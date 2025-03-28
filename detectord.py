"""
SpeciesNet Detector Service - Minimal Version for RunPod

A dedicated microservice for animal detection using the SpeciesNet detector model.
With lazy loading to minimize startup memory usage.
"""
import os
import base64
import tempfile
import time
import json
import logging
import multiprocessing
import socket
from flask import Flask, request, jsonify, abort, Blueprint

# Set up logging
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("detectord")

# Configure GPU environment variables
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"           # Reduce TensorFlow logging
os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"   # Prevent TF from grabbing all GPU memory
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"     # Match CUDA device IDs to hardware order

# Get environment variables
USE_GPU = os.getenv("USE_GPU", "true").lower() == "true"
INIT_AT_STARTUP = os.getenv("INIT_AT_STARTUP", "false").lower() == "true"

# Configure GPU environment variables
if not USE_GPU:
    logger.info("Disabling PyTorch GPU via environment variable")
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

# Fix multiprocessing hostname issues (critical fix for Docker)
hostname = socket.gethostname()
os.environ['HOSTNAME'] = hostname
os.environ['MULTIPROCESSING_AUTHKEY'] = "speciesnet_secret_key"

# Use spawn method for multiprocessing
if multiprocessing.current_process().name == 'MainProcess':
    try:
        multiprocessing.set_start_method('spawn', force=True)
        logger.info("Set multiprocessing start method to 'spawn'")
    except RuntimeError as e:
        logger.warning(f"Could not set multiprocessing start method: {e}")

# Global state
_detector = None
_start_time = time.time()
_initialization_error = None
gpu_info = {"pytorch": {"available": False, "device_count": 0, "error": None}}

def detect_gpus():
    """Detect available PyTorch GPUs"""
    try:
        import torch
        gpu_info["pytorch"]["available"] = torch.cuda.is_available()
        gpu_info["pytorch"]["device_count"] = torch.cuda.device_count() if torch.cuda.is_available() else 0
        if torch.cuda.is_available():
            gpu_info["pytorch"]["device_name"] = torch.cuda.get_device_name(0)
    except Exception as e:
        gpu_info["pytorch"]["error"] = str(e)
        logger.warning(f"Error checking PyTorch GPU: {e}")
    
    logger.info(f"GPU detection results: {gpu_info}")
    return gpu_info

# Create Flask app
app = Flask(__name__)

# Create health blueprint
health_bp = Blueprint('health', __name__)

@health_bp.route("/health", methods=["GET"])
def health():
    """Health check endpoint"""
    uptime = time.time() - _start_time
    model_status = {"initialized": _detector is not None}
    if _initialization_error:
        model_status["error"] = _initialization_error
    
    health_data = {
        "status": "healthy",
        "service": "detectord",
        "uptime_seconds": round(uptime, 2),
        "gpu_info": gpu_info,
        "model_status": model_status
    }
    
    # Add speciesnet info if available
    try:
        import speciesnet
        health_data["speciesnet_info"] = {
            "version": getattr(speciesnet, "__version__", "unknown")
        }
    except ImportError:
        health_data["speciesnet_info"] = "not installed"
    
    return jsonify(health_data)

@health_bp.route("/ready", methods=["GET"])
def ready():
    """Readiness check endpoint"""
    is_ready = _detector is not None
    if is_ready:
        return jsonify({
            "ready": True,
            "service": "detectord"
        })
    else:
        message = "Model not initialized yet"
        if _initialization_error:
            message = f"Model initialization failed: {_initialization_error}"
        return jsonify({
            "ready": False,
            "service": "detectord",
            "message": message
        }), 503  # Service Unavailable

app.register_blueprint(health_bp)

def get_detector():
    """Initialize the detector model if needed and return it"""
    global _detector, _initialization_error, gpu_info
    if _detector is None:
        logger.info("Initializing detector...")
        try:
            # Lazy import to reduce startup memory
            from speciesnet import SpeciesNet
            
            # Initialize detector
            _detector = SpeciesNet(
                model_name="kaggle:google/speciesnet/keras/v4.0.0a", 
                components="detector", 
                multiprocessing=False
            )
            logger.info(f"Detector initialized with PyTorch GPU: {gpu_info['pytorch']['available']}")
            _initialization_error = None
        except Exception as e:
            _initialization_error = str(e)
            logger.warning(f"Error loading detector: {e}")
            logger.warning("Attempting CPU-only mode")
            
            # Force CPU mode
            old_cuda_visible = os.environ.get('CUDA_VISIBLE_DEVICES', None)
            os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
            
            try:
                # Lazy import
                from speciesnet import SpeciesNet
                _detector = SpeciesNet(
                    model_name="kaggle:google/speciesnet/keras/v4.0.0a", 
                    components="detector", 
                    multiprocessing=False
                )
                logger.info("Detector initialized in CPU-only mode")
                _initialization_error = None
            except Exception as e2:
                _initialization_error = f"Failed in both GPU and CPU modes: {str(e2)}"
                logger.error(f"Failed in CPU mode: {e2}")
                raise
            finally:
                # Restore GPU settings
                if old_cuda_visible is not None:
                    os.environ['CUDA_VISIBLE_DEVICES'] = old_cuda_visible
    return _detector

# Create shared temp directory
TEMP_DIR = tempfile.mkdtemp(dir=os.getenv("SHARED_TEMP_DIR", "/tmp/shared_temp"))
logger.info(f"Created temp directory: {TEMP_DIR}")

@app.route("/detect", methods=["POST"])
def detect():
    """Endpoint to run animal detection on images"""
    try:
        # Get request data
        data = request.get_json()
        if not data or "instances" not in data:
            abort(400, description="Request must contain an 'instances' array")

        instances = data["instances"]
        if not isinstance(instances, list):
            abort(400, description="'instances' must be a list")

        # Prepare payload
        speciesnet_payload = {"instances": []}
        temp_files = []

        for instance in instances:
            if "image" not in instance:
                abort(400, description="Each instance must contain an 'image' key")

            # Decode base64 image
            base64_string = instance["image"]
            try:
                image_data = base64.b64decode(base64_string)
            except Exception as e:
                abort(400, description=f"Invalid base64 data: {str(e)}")
            
            # Save to temp file
            temp_file = tempfile.NamedTemporaryFile(
                delete=False, dir=TEMP_DIR, suffix=".jpg"
            )
            temp_files.append(temp_file.name)            
            
            with open(temp_file.name, "wb") as f:
                f.write(image_data)

            # Create instance for SpeciesNet
            speciesnet_instance = {"filepath": temp_file.name}
            # Copy metadata
            for key in instance:
                if key != "image":
                    speciesnet_instance[key] = instance[key]

            speciesnet_payload["instances"].append(speciesnet_instance)

        # Process with SpeciesNet
        try:
            detector = get_detector()
            result = detector.detect(
                instances_dict=speciesnet_payload, 
                run_mode='single_process',
                progress_bars=False
            )
            
            # Remove filepaths from results
            for p in result["predictions"]:
                if "filepath" in p:
                    del p["filepath"]

        except Exception as e:
            abort(500, description=f"Detection error: {str(e)}")

        # Clean up
        for temp_file in temp_files:
            try:
                os.remove(temp_file)
            except OSError:
                pass
                
        return jsonify(result)

    except Exception as e:
        logger.exception("Error in detect endpoint")
        # Clean up on error
        for temp_file in temp_files:
            try:
                os.remove(temp_file)
            except OSError:
                pass
        abort(500, description=f"Server error: {str(e)}")

# Detect GPUs at startup
detect_gpus()

# Initialize detector at startup if configured
if INIT_AT_STARTUP:
    try:
        get_detector()
        logger.info("Pre-initialization complete")
    except Exception as e:
        logger.error(f"Pre-initialization failed: {e}")

if __name__ == "__main__":
    port = int(os.getenv("PORT", 5001))
    logger.info(f"Starting detector service on port {port}, GPU={USE_GPU}")
    app.run(host="0.0.0.0", port=port, debug=False)