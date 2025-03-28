"""
SpeciesNet Classifier Service

A dedicated microservice for species classification using the SpeciesNet classifier model.
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
logger = setup_logging("classiferd")

# Configure GPU settings
configure_gpu() 

# Get environment variables
USE_GPU = os.getenv("USE_GPU", "true").lower() == "true"
INIT_AT_STARTUP = os.getenv("INIT_AT_STARTUP", "true").lower() == "true"

# Set multiprocessing method
setup_multiprocessing(logger, use_gpu=USE_GPU)

# Define GPU detection
gpu_info = {"tensorflow": {"available": False, "device_count": 0, "error": None}}

def detect_gpus():
    """Detect available TensorFlow GPUs"""
    try:
        import tensorflow as tf
        gpus = tf.config.list_physical_devices('GPU')
        gpu_info["tensorflow"]["available"] = len(gpus) > 0
        gpu_info["tensorflow"]["device_count"] = len(gpus)
        
        # Configure TensorFlow to use GPU efficiently
        if len(gpus) > 0:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
    except Exception as e:
        gpu_info["tensorflow"]["error"] = str(e)
        logger.warning(f"Error checking TensorFlow GPU: {e}")
    
    logger.info(f"GPU detection results: {gpu_info}")
    return gpu_info

# Configure GPU environment variables
if not USE_GPU:
    logger.info("Disabling TensorFlow GPU via environment variable")
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"

# Detect GPUs at startup
detect_gpus()

# Import SpeciesNet after GPU configuration
from speciesnet import SpeciesNet

# Global state
_classifier = None
_start_time = time.time()
_initialization_error = None

def get_classifier() -> SpeciesNet:
    """Initialize the classifier model if needed and return it"""
    global _classifier, _initialization_error
    if (_classifier is None):
        logger.info("Initializing classifier...")
        try:
            _classifier = SpeciesNet(model_name="kaggle:google/speciesnet/keras/v4.0.0a", 
                                  components="classifier", 
                                  multiprocessing=True)
            logger.info(f"Classifier initialized with TensorFlow GPU available: {gpu_info['tensorflow']['available']}")
            _initialization_error = None
        except Exception as e:
            _initialization_error = str(e)
            logger.error(f"Failed to initialize classifier: {e}")
            # For TensorFlow, we don't have a simple CPU fallback like PyTorch
            # It should already work with CPU if GPU isn't available
            raise
    return _classifier

# Create Flask app
app = Flask(__name__)

# Add health endpoints
app.register_blueprint(create_health_blueprint(
    service_name="classifierd", 
    start_time=_start_time,
    model=_classifier,
    initialization_error=_initialization_error,
    gpu_info=gpu_info
))

# Create temporary directory for file operations
TEMP_DIR = tempfile.mkdtemp(dir=os.getenv("SHARED_TEMP_DIR", "/tmp/shared_temp"))
logger.info(f"Created temp directory: {TEMP_DIR}")

@app.route("/classify", methods=["POST"])
def classify():
    """Endpoint to run species classification on images"""
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
            classifier = get_classifier()
            speciesnet_result = classifier.classify(
                instances_dict=speciesnet_payload, 
                run_mode='multi_process',
                progress_bars=False, 
                predictions_json=None
            )
            
            # Remove temporary filepaths from results
            for p in speciesnet_result["predictions"]:
                if "filepath" in p:
                    del p["filepath"]

        except Exception as e:
            abort(500, description=f"Classification error: {str(e)}")

        # Clean up temporary files
        for temp_file in temp_files:
            try:
                os.remove(temp_file)
            except OSError:
                pass
                
        # Return results
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

# Initialize classifier at startup if configured
if INIT_AT_STARTUP:
    def init_classifier():
        try:
            get_classifier()
            logger.info("Pre-initialization complete")
        except Exception as e:
            logger.error(f"Pre-initialization failed: {e}")
    
    # Initialize in a background thread to not block server startup
    threading.Thread(target=init_classifier, daemon=True).start()

if __name__ == "__main__":
    port = int(os.getenv("PORT", 5002))
    logger.info(f"Starting classifier service on port {port}")
    logger.info(f"GPU enabled: {USE_GPU}")
    
    # Start Flask server
    app.run(host="0.0.0.0", port=port, debug=False)