import os
import base64
import tempfile
import shutil
import requests
import json
import logging
import time
import threading
from flask import Flask, request, jsonify, abort
from werkzeug.utils import secure_filename
import sys

# Set up logging
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Environment variables for GPU control
USE_GPU_DETECTOR = os.getenv("USE_GPU_DETECTOR", "auto").lower()  # "auto", "true", or "false"
USE_GPU_CLASSIFIER = os.getenv("USE_GPU_CLASSIFIER", "auto").lower()  # "auto", "true", or "false"

# Determine which components to initialize at startup
INIT_DETECTOR = os.getenv("INIT_DETECTOR", "true").lower() == "true"
INIT_CLASSIFIER = os.getenv("INIT_CLASSIFIER", "true").lower() == "true"
INIT_ENSEMBLE = os.getenv("INIT_ENSEMBLE", "false").lower() == "true"

# Add this code after your existing import statements
# Determine operating mode based on GPU settings
if USE_GPU_DETECTOR == "true" and USE_GPU_CLASSIFIER != "true":
    OPERATING_MODE = "detector"
    logger.info("Running in DETECTOR mode")
elif USE_GPU_CLASSIFIER == "true" and USE_GPU_DETECTOR != "true":
    OPERATING_MODE = "classifier"
    logger.info("Running in CLASSIFIER mode")
else:
    OPERATING_MODE = "dual"
    logger.info("Running in DUAL mode (both detector and classifier)")

# Update initialization settings based on operating mode
if OPERATING_MODE == "detector":
    INIT_DETECTOR = True
    INIT_CLASSIFIER = False
    INIT_ENSEMBLE = False
elif OPERATING_MODE == "classifier":
    INIT_DETECTOR = False
    INIT_CLASSIFIER = True
    INIT_ENSEMBLE = False
else:
    # For dual mode, use the environment variables as-is
    INIT_DETECTOR = os.getenv("INIT_DETECTOR", "true").lower() == "true"
    INIT_CLASSIFIER = os.getenv("INIT_CLASSIFIER", "true").lower() == "true"
    INIT_ENSEMBLE = os.getenv("INIT_ENSEMBLE", "false").lower() == "true"

# GPU detection results
gpu_info = {
    "pytorch": {"available": False, "device_count": 0, "error": None},
    "tensorflow": {"available": False, "device_count": 0, "error": None}
}

# Detect available GPUs
def detect_gpus():
    # Check PyTorch GPUs
    try:
        import torch
        gpu_info["pytorch"]["available"] = torch.cuda.is_available()
        gpu_info["pytorch"]["device_count"] = torch.cuda.device_count() if torch.cuda.is_available() else 0
        if torch.cuda.is_available():
            gpu_info["pytorch"]["device_name"] = torch.cuda.get_device_name(0)
    except Exception as e:
        gpu_info["pytorch"]["error"] = str(e)
        logger.warning(f"Error checking PyTorch GPU: {e}")

    # Check TensorFlow GPUs
    try:
        import tensorflow as tf
        gpus = tf.config.list_physical_devices('GPU')
        gpu_info["tensorflow"]["available"] = len(gpus) > 0
        gpu_info["tensorflow"]["device_count"] = len(gpus)
        
        # If TensorFlow sees GPUs but PyTorch should use them
        if len(gpus) > 0 and USE_GPU_DETECTOR == "true" and USE_GPU_CLASSIFIER != "true":
            logger.info("Restricting TensorFlow GPU access to allow PyTorch to use GPUs")
            # Prevent TensorFlow from using all GPU memory
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            if USE_GPU_CLASSIFIER == "false":
                logger.info("Disabling TensorFlow GPU access completely")
                tf.config.set_visible_devices([], 'GPU')
                
    except Exception as e:
        gpu_info["tensorflow"]["error"] = str(e)
        logger.warning(f"Error checking TensorFlow GPU: {e}")
    
    logger.info(f"GPU detection results: {gpu_info}")
    return gpu_info

# Detect GPUs at startup
detect_gpus()

# Configure GPU environment variables based on settings
if USE_GPU_DETECTOR == "false":
    # Disable PyTorch GPU
    logger.info("Disabling PyTorch/detector GPU via environment variable")
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

if USE_GPU_CLASSIFIER == "false":
    # Disable TensorFlow GPU
    logger.info("Disabling TensorFlow/classifier GPU via environment variable")
    if "CUDA_VISIBLE_DEVICES" not in os.environ:
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"

# Import SpeciesNet after GPU configuration
from speciesnet import SpeciesNet

# Shared states
_detector = None
_classifier = None
_ensemble = None
_start_time = time.time()
_initialization_errors = {
    "detector": None,
    "classifier": None,
    "ensemble": None
}

def get_detector() -> SpeciesNet:
    global _detector, _initialization_errors
    if (_detector is None):
        logger.info("Initializing detector...")
        try:
            # First attempt with standard initialization
            _detector = SpeciesNet(model_name="kaggle:google/speciesnet/keras/v4.0.0a", 
                                 components="detector", 
                                 multiprocessing=True)
            logger.info(f"Detector initialized with PyTorch GPU available: {gpu_info['pytorch']['available']}")
        except Exception as e:
            _initialization_errors["detector"] = str(e)
            if any(err_msg in str(e) for err_msg in ["CUDA", "cuda", "device"]):
                # Handle CUDA issues by forcing CPU mode
                logger.warning(f"CUDA error encountered when loading detector: {e}")
                logger.warning("Attempting to load detector in CPU-only mode")
                
                # Force CPU mode for PyTorch temporarily
                old_cuda_visible = os.environ.get('CUDA_VISIBLE_DEVICES', None)
                os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
                
                try:
                    _detector = SpeciesNet(model_name="kaggle:google/speciesnet/keras/v4.0.0a", 
                                         components="detector", 
                                         multiprocessing=False)  # Disable multiprocessing to avoid shared CUDA contexts
                    logger.info("Detector initialized in CPU-only fallback mode")
                    _initialization_errors["detector"] = None  # Clear the error since we succeeded in CPU mode
                except Exception as e2:
                    _initialization_errors["detector"] = f"Failed in both GPU and CPU modes: {str(e2)}"
                    logger.error(f"Failed to initialize detector in CPU mode: {e2}")
                    raise
                finally:
                    # Restore original CUDA settings
                    if old_cuda_visible is not None:
                        os.environ['CUDA_VISIBLE_DEVICES'] = old_cuda_visible
                    elif 'CUDA_VISIBLE_DEVICES' in os.environ:
                        del os.environ['CUDA_VISIBLE_DEVICES']
            else:
                # Re-raise if not a CUDA issue
                raise
    return _detector

def get_classifier() -> SpeciesNet:
    global _classifier, _initialization_errors
    if (_classifier is None):
        logger.info("Initializing classifier...")
        try:
            _classifier = SpeciesNet(model_name="kaggle:google/speciesnet/keras/v4.0.0a", 
                                  components="classifier", 
                                  multiprocessing=True)
            logger.info(f"Classifier initialized with TensorFlow GPU available: {gpu_info['tensorflow']['available']}")
        except Exception as e:
            _initialization_errors["classifier"] = str(e)
            logger.error(f"Failed to initialize classifier: {e}")
            raise
    return _classifier

def get_ensemble() -> SpeciesNet:
    global _ensemble, _initialization_errors
    if (_ensemble is None):
        logger.info("Initializing ensemble...")
        try:
            _ensemble = SpeciesNet(model_name="kaggle:google/speciesnet/keras/v4.0.0a", 
                               components="ensemble", 
                               multiprocessing=True)
            logger.info(f"Ensemble initialized with PyTorch GPU: {gpu_info['pytorch']['available']}, TensorFlow GPU: {gpu_info['tensorflow']['available']}")
        except Exception as e:
            _initialization_errors["ensemble"] = str(e)
            logger.error(f"Failed to initialize ensemble: {e}")
            raise
    return _ensemble

app = Flask(__name__)

# Configuration
# Specify the parent directory (must exist and be writable)
# recommend this be an in memory temp dir
SPECIFIC_PATH = os.getenv("SHARED_TEMP_DIR", "/tmp/shared_temp")

LISTEN_PORT = int(os.getenv("LISTEN_PORT", 5100))  # port that the api listens to

# Ensure the parent directory exists
os.makedirs(SPECIFIC_PATH, exist_ok=True)

# Create a temporary directory under the specific path
TEMP_DIR = tempfile.mkdtemp(dir=SPECIFIC_PATH)
logger.info(f"Created temp directory: {TEMP_DIR}")

@app.route("/health", methods=["GET"])
def health():
    """Health check endpoint for monitoring and container orchestration."""
    uptime = time.time() - _start_time
    
    # Check if components have been initialized
    detector_status = {"initialized": _detector is not None}
    classifier_status = {"initialized": _classifier is not None}
    ensemble_status = {"initialized": _ensemble is not None}
    
    # Add detailed error information if components have failed to initialize
    if _initialization_errors["detector"]:
        detector_status["error"] = _initialization_errors["detector"]
    
    if _initialization_errors["classifier"]:
        classifier_status["error"] = _initialization_errors["classifier"]
    
    if _initialization_errors["ensemble"]:
        ensemble_status["error"] = _initialization_errors["ensemble"]
    
    # Try to get GPU memory info if available
    gpu_memory_info = {}
    try:
        import torch
        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                mem_allocated = round(torch.cuda.memory_allocated(i) / (1024**2), 2)  # MB
                mem_reserved = round(torch.cuda.memory_reserved(i) / (1024**2), 2)    # MB
                gpu_memory_info[f"gpu_{i}"] = {
                    "allocated_mb": mem_allocated,
                    "reserved_mb": mem_reserved,
                    "name": torch.cuda.get_device_name(i)
                }
    except Exception as e:
        gpu_memory_info["error"] = str(e)
    
    # Basic health check - we're alive
    health_data = {
        "status": "healthy",
        "operating_mode": OPERATING_MODE,
        "uptime_seconds": round(uptime, 2),
        "temp_directory": os.path.exists(TEMP_DIR),
        "gpu_info": gpu_info,
        "gpu_memory": gpu_memory_info,
        "gpu_settings": {
            "detector": USE_GPU_DETECTOR,
            "classifier": USE_GPU_CLASSIFIER
        },
        "components": {
            "detector": detector_status,
            "classifier": classifier_status,
            "ensemble": ensemble_status
        },
        "python_version": sys.version,
        "environment": {
            "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES", "not set"),
            "TF_FORCE_GPU_ALLOW_GROWTH": os.environ.get("TF_FORCE_GPU_ALLOW_GROWTH", "not set")
        },
        "initialization_settings": {
            "init_detector": INIT_DETECTOR,
            "init_classifier": INIT_CLASSIFIER,
            "init_ensemble": INIT_ENSEMBLE
        }
    }
    
    # Add version info if available
    try:
        import speciesnet
        health_data["speciesnet_info"] = {
            "module_path": speciesnet.__file__,
            "version": getattr(speciesnet, "__version__", "unknown")
        }
        
        # Add pytorch/tensorflow version info
        try:
            import torch
            health_data["torch_version"] = torch.__version__
        except ImportError:
            pass
            
        try:
            import tensorflow as tf
            health_data["tensorflow_version"] = tf.__version__
        except ImportError:
            pass
    except (ImportError, AttributeError):
        health_data["speciesnet_info"] = None
        
    return jsonify(health_data)

# Add a readiness endpoint for container orchestration systems
@app.route("/ready", methods=["GET"])
def ready():
    """Readiness check endpoint that verifies if all components are loaded."""
    # Check readiness based on operating mode
    if OPERATING_MODE == "detector":
        ready = _detector is not None
        required_components = ["detector"]
    elif OPERATING_MODE == "classifier":
        ready = _classifier is not None
        required_components = ["classifier"]
    else:  # dual mode
        # We'll consider the service ready if all required components are initialized
        required_components = []
        if INIT_DETECTOR:
            required_components.append("detector")
        if INIT_CLASSIFIER:
            required_components.append("classifier")
        if INIT_ENSEMBLE:
            required_components.append("ensemble")
            
        # Check if all required components are initialized
        ready = all([
            (_detector is not None) if "detector" in required_components else True,
            (_classifier is not None) if "classifier" in required_components else True,
            (_ensemble is not None) if "ensemble" in required_components else True
        ])
    
    if ready:
        return jsonify({
            "ready": True,
            "operating_mode": OPERATING_MODE,
            "required_components": required_components,
            "components": {
                "detector": _detector is not None,
                "classifier": _classifier is not None,
                "ensemble": _ensemble is not None
            }
        })
    else:
        return jsonify({
            "ready": False,
            "operating_mode": OPERATING_MODE,
            "message": f"Not all required components for {OPERATING_MODE} mode have been initialized yet",
            "required_components": required_components,
            "components": {
                "detector": _detector is not None,
                "classifier": _classifier is not None,
                "ensemble": _ensemble is not None
            }
        }), 503  # Service Unavailable

@app.route("/ensemble", methods=["POST"])
def ensemble():
    return jsonify({"message": "Ensemble endpoint is not implemented yet."})

@app.route("/classify", methods=["POST"])
def classify():
    # Check if we're in detector mode (not allowed to classify)
    if OPERATING_MODE == "detector":
        abort(405, description="This instance is running in DETECTOR mode and cannot process classification requests. Please use a CLASSIFIER instance instead.")
        
    try:
        # Get JSON data from the request
        data = request.get_json()
        if not data or "instances" not in data:
            abort(400, description="Request must contain an 'instances' array")

        instances = data["instances"]
        if not isinstance(instances, list):
            abort(400, description="'instances' must be a list")

        # Prepare payload for SpeciesNet API
        speciesnet_payload = {"instances": []}
        temp_files = []  # Track files to clean up later

        for instance in instances:
            if "image" not in instance:
                abort(400, description="Each instance must contain an 'image' key with base64 data")

            # Decode base64 image
            base64_string = instance["image"]
            try:
                image_data = base64.b64decode(base64_string)
            except Exception as e:
                abort(400, description=f"Invalid base64 data: {str(e)}")
            
            # Save to a temporary file
            temp_file = tempfile.NamedTemporaryFile(
                delete=False, dir=TEMP_DIR, suffix=".jpg"
            )
            temp_files.append(temp_file.name)            
            
            with open(temp_file.name, "wb") as f:
                f.write(image_data)

            # Create instance for SpeciesNet with filepath
            speciesnet_instance = {
                "filepath": temp_file.name
            }
            # Copy other metadata (e.g., country, admin1_region) unchanged
            for key in instance:
                if key != "image":
                    speciesnet_instance[key] = instance[key]

            speciesnet_payload["instances"].append(speciesnet_instance)

        # Send request to SpeciesNet API
        try:
            speciesnet_result = get_classifier().classify(instances_dict=speciesnet_payload, run_mode='multi_process', progress_bars=False, predictions_json=None)
            # Since filepath is always a tmpfile we can remove it from the speciesnet_result
            for p in speciesnet_result["predictions"]:
                if "filepath" in p:
                    del p["filepath"]

        except requests.exceptions.RequestException as e:
            abort(500, description=f"Failed to forward request to SpeciesNet API: {str(e)}")

        # Return SpeciesNet response
        # cleanup temp file
        for temp_file in temp_files:
            try:
                os.remove(temp_file)
            except OSError:
                pass

        return jsonify(speciesnet_result)

    except Exception as e:
        logger.exception("Error in classify endpoint")
        # Clean up files in case of error
        for temp_file in temp_files:
            try:
                os.remove(temp_file)
            except OSError:
                pass
        abort(500, description=f"Server error: {str(e)}")


@app.route("/detect", methods=["POST"])
def detect():
    # Check if we're in classifier mode (not allowed to detect)
    if OPERATING_MODE == "classifier":
        abort(405, description="This instance is running in CLASSIFIER mode and cannot process detection requests. Please use a DETECTOR instance instead.")
        
    try:
        # Get JSON data from the request
        data = request.get_json()
        if not data or "instances" not in data:
            abort(400, description="Request must contain an 'instances' array")

        instances = data["instances"]
        if not isinstance(instances, list):
            abort(400, description="'instances' must be a list")

        # Prepare payload for SpeciesNet API
        speciesnet_payload = {"instances": []}
        temp_files = []  # Track files to clean up later

        for instance in instances:
            if "image" not in instance:
                abort(400, description="Each instance must contain an 'image' key with base64 data")

            # Decode base64 image
            base64_string = instance["image"]
            try:
                image_data = base64.b64decode(base64_string)
            except Exception as e:
                abort(400, description=f"Invalid base64 data: {str(e)}")
            
            # Save to a temporary file
            temp_file = tempfile.NamedTemporaryFile(
                delete=False, dir=TEMP_DIR, suffix=".jpg"
            )
            temp_files.append(temp_file.name)            
            
            with open(temp_file.name, "wb") as f:
                f.write(image_data)

            # Create instance for SpeciesNet with filepath
            speciesnet_instance = {
                "filepath": temp_file.name
            }
            # Copy other metadata (e.g., country, admin1_region) unchanged
            for key in instance:
                if key != "image":
                    speciesnet_instance[key] = instance[key]

            speciesnet_payload["instances"].append(speciesnet_instance)

        # Send request to SpeciesNet API
        try:
            speciesnet_result = get_detector().detect(instances_dict=speciesnet_payload, run_mode='multi_process', progress_bars=False, predictions_json=None)
            # Since filepath is always a tmpfile we can remove it from the speciesnet_result
            for p in speciesnet_result["predictions"]:
                if "filepath" in p:
                    del p["filepath"]

        except requests.exceptions.RequestException as e:
            abort(500, description=f"Failed to forward request to SpeciesNet API: {str(e)}")

        # remove tempfiles
        for temp_file in temp_files:
            try:
                os.remove(temp_file)
            except OSError:
                pass
        # Return SpeciesNet response
        return jsonify(speciesnet_result)

    except Exception as e:
        logger.exception("Error in detect endpoint")
        # Clean up files in case of error
        for temp_file in temp_files:
            try:
                os.remove(temp_file)
            except OSError:
                pass
        abort(500, description=f"Server error: {str(e)}")

# Background initialization function
def initialize_components():
    """Initialize selected components in background threads based on operating mode"""
    logger.info(f"Initializing components for {OPERATING_MODE} mode")
    
    if OPERATING_MODE == "detector" or (OPERATING_MODE == "dual" and INIT_DETECTOR):
        logger.info("Pre-initializing detector in background thread")
        threading.Thread(target=lambda: get_detector(), daemon=True).start()
    
    if OPERATING_MODE == "classifier" or (OPERATING_MODE == "dual" and INIT_CLASSIFIER):
        logger.info("Pre-initializing classifier in background thread")
        threading.Thread(target=lambda: get_classifier(), daemon=True).start()
    
    if OPERATING_MODE == "dual" and INIT_ENSEMBLE:
        logger.info("Pre-initializing ensemble in background thread")
        threading.Thread(target=lambda: get_ensemble(), daemon=True).start()

if __name__ == "__main__":
    logger.info(f"Starting Flask server on port {LISTEN_PORT}")
    logger.info(f"GPU settings: detector={USE_GPU_DETECTOR}, classifier={USE_GPU_CLASSIFIER}")
    
    # Initialize components before starting the server
    initialize_components()
    
    # Start the Flask server
    app.run(host="0.0.0.0", port=LISTEN_PORT, debug=False)