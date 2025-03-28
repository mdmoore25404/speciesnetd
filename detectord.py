"""
SpeciesNet Detector Service

A dedicated microservice for animal detection using the SpeciesNet detector model.
"""
import os
import base64
import tempfile
import time
import threading
from flask import Flask, request, jsonify, abort

# Import shared utilities
from common import (setup_logging, configure_gpu, setup_multiprocessing, 
                   create_health_blueprint)

# Set up logger
logger = setup_logging("detectord")

# Configure GPU settings
configure_gpu()

# Get environment variables
USE_GPU = os.getenv("USE_GPU", "true").lower() == "true"
INIT_AT_STARTUP = os.getenv("INIT_AT_STARTUP", "true").lower() == "true"

# Set multiprocessing method
setup_multiprocessing(logger, use_gpu=USE_GPU)

# Define GPU detection
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

# Configure GPU environment variables
if not USE_GPU:
    logger.info("Disabling PyTorch GPU via environment variable")
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

# Detect GPUs at startup
detect_gpus()

# Import SpeciesNet after GPU configuration
from speciesnet import SpeciesNet

# Global state
_detector = None
_start_time = time.time()
_initialization_error = None

def get_detector() -> SpeciesNet:
    """Initialize the detector model if needed and return it"""
    global _detector, _initialization_error
    if (_detector is None):
        logger.info("Initializing detector...")
        try:
            # If GPU is requested but we're using 'fork' method, use multiprocessing=False
            use_multiprocessing = True
            if USE_GPU and multiprocessing.get_start_method(allow_none=True) == 'fork':
                logger.warning("CUDA with 'fork' multiprocessing detected; disabling multiprocessing")
                use_multiprocessing = False
                
            _detector = SpeciesNet(model_name="kaggle:google/speciesnet/keras/v4.0.0a", 
                                 components="detector", 
                                 multiprocessing=use_multiprocessing)
            logger.info(f"Detector initialized with PyTorch GPU available: {gpu_info['pytorch']['available']}")
            _initialization_error = None
        except Exception as e:
            _initialization_error = str(e)
            # Check for CUDA errors or multiprocessing errors
            if any(err_msg in str(e) for err_msg in ["CUDA", "cuda", "device", "subprocess", "multiprocessing"]):
                logger.warning(f"CUDA/multiprocessing error encountered: {e}")
                logger.warning("Attempting to load detector in CPU-only mode")
                
                # Force CPU mode temporarily
                old_cuda_visible = os.environ.get('CUDA_VISIBLE_DEVICES', None)
                os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
                
                try:
                    _detector = SpeciesNet(model_name="kaggle:google/speciesnet/keras/v4.0.0a", 
                                         components="detector", 
                                         multiprocessing=False)
                    logger.info("Detector initialized in CPU-only fallback mode")
                    _initialization_error = None
                except Exception as e2:
                    _initialization_error = f"Failed in both GPU and CPU modes: {str(e2)}"
                    logger.error(f"Failed to initialize detector in CPU mode: {e2}")
                    raise
                finally:
                    # Restore original CUDA settings
                    if old_cuda_visible is not None:
                        os.environ['CUDA_VISIBLE_DEVICES'] = old_cuda_visible
                    elif 'CUDA_VISIBLE_DEVICES' in os.environ:
                        del os.environ['CUDA_VISIBLE_DEVICES']
            else:
                logger.error(f"Failed to initialize detector: {e}")
                raise
    return _detector

# Create Flask app
app = Flask(__name__)

# Add health endpoints
app.register_blueprint(create_health_blueprint(
    service_name="detectord", 
    start_time=_start_time,
    model=_detector,
    initialization_error=_initialization_error,
    gpu_info=gpu_info
))

# Create temporary directory for file operations
TEMP_DIR = tempfile.mkdtemp(dir=os.getenv("SHARED_TEMP_DIR", "/tmp/shared_temp"))
logger.info(f"Created temp directory: {TEMP_DIR}")

@app.route("/detect", methods=["POST"])
def detect():
    """Endpoint to run animal detection on images"""
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
            # Copy other metadata (e.g., country, admin1_region)
            for key in instance:
                if key != "image":
                    speciesnet_instance[key] = instance[key]

            speciesnet_payload["instances"].append(speciesnet_instance)

        # Process with SpeciesNet
        try:
            detector = get_detector()
            speciesnet_result = detector.detect(
                instances_dict=speciesnet_payload, 
                run_mode='multi_process' if USE_GPU else 'single_process',
                progress_bars=False, 
                predictions_json=None
            )
            
            # Remove temporary filepaths from results
            for p in speciesnet_result["predictions"]:
                if "filepath" in p:
                    del p["filepath"]

        except Exception as e:
            abort(500, description=f"Detection error: {str(e)}")

        # Clean up temporary files
        for temp_file in temp_files:
            try:
                os.remove(temp_file)
            except OSError:
                pass
                
        # Return results
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

# Initialize detector at startup if configured
if INIT_AT_STARTUP:
    def init_detector():
        try:
            get_detector()
            logger.info("Pre-initialization complete")
        except Exception as e:
            logger.error(f"Pre-initialization failed: {e}")
    
    # Initialize in a background thread to not block server startup
    threading.Thread(target=init_detector, daemon=True).start()

if __name__ == "__main__":
    port = int(os.getenv("PORT", 5001))
    logger.info(f"Starting detector service on port {port}")
    logger.info(f"GPU enabled: {USE_GPU}")
    
    # Start Flask server
    app.run(host="0.0.0.0", port=port, debug=False)