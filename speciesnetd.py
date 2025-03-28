import os
import base64
import tempfile
import shutil
import requests
import json
import logging
import time
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

# Lazy initialization of SpeciesNet objects
_detector = None
_classifier = None
_ensemble = None
_start_time = time.time()

def get_detector() -> SpeciesNet:
    global _detector
    if (_detector is None):
        logger.info("Initializing detector...")
        # FIXED: Remove use_gpu parameter
        _detector = SpeciesNet(model_name="kaggle:google/speciesnet/keras/v4.0.0a", 
                             components="detector", 
                             multiprocessing=True)
        logger.info(f"Detector initialized with PyTorch GPU available: {gpu_info['pytorch']['available']}")
    return _detector

def get_classifier() -> SpeciesNet:
    global _classifier
    if (_classifier is None):
        logger.info("Initializing classifier...")
        # FIXED: Remove use_gpu parameter
        _classifier = SpeciesNet(model_name="kaggle:google/speciesnet/keras/v4.0.0a", 
                               components="classifier", 
                               multiprocessing=True)
        logger.info(f"Classifier initialized with TensorFlow GPU available: {gpu_info['tensorflow']['available']}")
    return _classifier

def get_ensemble() -> SpeciesNet:
    global _ensemble
    if (_ensemble is None):
        logger.info("Initializing ensemble...")
        # FIXED: Remove use_gpu parameter
        _ensemble = SpeciesNet(model_name="kaggle:google/speciesnet/keras/v4.0.0a", 
                            components="ensemble", 
                            multiprocessing=True)
        logger.info(f"Ensemble initialized with PyTorch GPU: {gpu_info['pytorch']['available']}, TensorFlow GPU: {gpu_info['tensorflow']['available']}")
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
    """Health check endpoint for monitoring and container orchestration.
    
    Returns:
        JSON: Health status information including uptime and component status
    """
    uptime = time.time() - _start_time
    
    # Basic health check - we're alive
    health_data = {
        "status": "healthy",
        "uptime_seconds": round(uptime, 2),
        "temp_directory": os.path.exists(TEMP_DIR),
        "gpu_info": gpu_info,
        "gpu_settings": {
            "detector": USE_GPU_DETECTOR,
            "classifier": USE_GPU_CLASSIFIER
        },
        "components": {
            "detector": _detector is not None,
            "classifier": _classifier is not None,
            "ensemble": _ensemble is not None
        },
        "python_version": sys.version,
        "memory_info": {
            "available": "Unknown"  # We'll add memory info in a future enhancement
        }
    }
    
    # Add version info if available
    try:
        import speciesnet
        health_data["speciesnet_info"] = {
            "module_path": speciesnet.__file__
        }
    except (ImportError, AttributeError):
        health_data["speciesnet_info"] = None
        
    return jsonify(health_data)

# Add a readiness endpoint for container orchestration systems
@app.route("/ready", methods=["GET"])
def ready():
    """Readiness check endpoint that verifies if all components are loaded.
    
    Returns:
        JSON: Readiness status with component information
    """
    # We'll consider the service ready if at least one of the models is initialized
    components_ready = any([_detector is not None, _classifier is not None, _ensemble is not None])
    
    if components_ready:
        return jsonify({
            "ready": True,
            "components": {
                "detector": _detector is not None,
                "classifier": _classifier is not None,
                "ensemble": _ensemble is not None
            }
        })
    else:
        return jsonify({
            "ready": False,
            "message": "No SpeciesNet components have been initialized yet"
        }), 503  # Service Unavailable

@app.route("/ensemble", methods=["POST"])
def ensemble():
    return jsonify({"message": "Ensemble endpoint is not implemented yet."})

@app.route("/classify", methods=["POST"])
def classify():
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

if __name__ == "__main__":
    logger.info(f"Starting Flask server on port {LISTEN_PORT}")
    logger.info(f"GPU settings: detector={USE_GPU_DETECTOR}, classifier={USE_GPU_CLASSIFIER}")
    app.run(host="0.0.0.0", port=LISTEN_PORT, debug=False)