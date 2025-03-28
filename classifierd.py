"""
SpeciesNet Classifier Service - Minimal Version for RunPod

A dedicated microservice for species classification using the SpeciesNet classifier model.
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
logger = logging.getLogger("classifierd")

# Configure GPU environment variables
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "1"           # Only show warnings and errors, not info
os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"   # Prevent TF from grabbing all GPU memory
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"     # Match CUDA device IDs to hardware order

# Get environment variables
USE_GPU = os.getenv("USE_GPU", "true").lower() == "true"
INIT_AT_STARTUP = os.getenv("INIT_AT_STARTUP", "false").lower() == "true"

# Configure GPU environment variables
if not USE_GPU:
    logger.info("Disabling TensorFlow GPU via environment variable")
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
else:
    # Explicitly enable GPU
    logger.info("Explicitly enabling TensorFlow GPU")
    # Make sure this environment variable is not set to -1
    if os.environ.get("CUDA_VISIBLE_DEVICES") == "-1":
        del os.environ["CUDA_VISIBLE_DEVICES"]
    # Force TensorFlow to use GPU
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"

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
_classifier = None
_start_time = time.time()
_initialization_error = None
gpu_info = {"tensorflow": {"available": False, "device_count": 0, "error": None}}

def detect_gpus():
    """Detect available TensorFlow GPUs"""
    try:
        # Check CUDA environment variables
        logger.info("CUDA Environment Variables:")
        for env_var in ['CUDA_VISIBLE_DEVICES', 'LD_LIBRARY_PATH', 'NVIDIA_DRIVER_CAPABILITIES', 
                        'NVIDIA_VISIBLE_DEVICES', 'CUDA_HOME', 'TF_FORCE_GPU_ALLOW_GROWTH']:
            logger.info(f"  {env_var}: {os.environ.get(env_var, 'not set')}")
        
        # Try to directly check NVIDIA GPUs
        try:
            import subprocess
            result = subprocess.run(["nvidia-smi"], stdout=subprocess.PIPE, text=True)
            logger.info(f"nvidia-smi output:\n{result.stdout}")
        except Exception as e:
            logger.warning(f"Failed to run nvidia-smi: {e}")
        
        import tensorflow as tf
        # Print TensorFlow version for debugging
        logger.info(f"TensorFlow version: {tf.__version__}")
        
        # Disable verbose device placement logging
        tf.debugging.set_log_device_placement(False)  # Change to False
        
        # Get TensorFlow GPU info
        gpus = tf.config.list_physical_devices('GPU')
        logger.info(f"TensorFlow physical GPUs: {gpus}")
        
        gpu_info["tensorflow"]["available"] = len(gpus) > 0
        gpu_info["tensorflow"]["device_count"] = len(gpus)
        
        # Get device names if available
        if gpus:
            # Enable memory growth to prevent TF from grabbing all memory
            for gpu in gpus:
                try:
                    tf.config.experimental.set_memory_growth(gpu, True)
                    logger.info(f"Enabled memory growth for GPU {gpu}")
                except Exception as e:
                    logger.warning(f"Could not set memory growth for GPU {gpu}: {e}")
                    
            device_details = []
            for gpu in gpus:
                try:
                    # Get device details
                    details = tf.config.experimental.get_device_details(gpu)
                    device_details.append(details.get('device_name', str(gpu)))
                except:
                    device_details.append(str(gpu))
            gpu_info["tensorflow"]["device_names"] = device_details
            
            # Test GPU with a simple computation
            with tf.device('/GPU:0'):
                a = tf.constant([[1.0, 2.0], [3.0, 4.0]])
                b = tf.constant([[1.0, 1.0], [1.0, 1.0]])
                c = tf.matmul(a, b)
                logger.info(f"Simple GPU test result: {c}")
                gpu_info["tensorflow"]["test_passed"] = True
    except Exception as e:
        gpu_info["tensorflow"]["error"] = str(e)
        logger.warning(f"Error checking TensorFlow GPU: {e}")
    
    logger.info(f"GPU detection results: {gpu_info}")
    return gpu_info

def force_gpu_test():
    """Last resort attempt to use GPU"""
    try:
        import tensorflow as tf
        import numpy as np
        
        # Create a small test model
        logger.info("Creating test model to verify GPU usage")
        inputs = tf.keras.layers.Input(shape=(224, 224, 3))
        x = tf.keras.layers.Conv2D(16, 3, activation='relu')(inputs)
        x = tf.keras.layers.GlobalAveragePooling2D()(x)
        outputs = tf.keras.layers.Dense(10, activation='softmax')(x)
        model = tf.keras.Model(inputs=inputs, outputs=outputs)
        
        # Compile and test
        model.compile(optimizer='adam', loss='categorical_crossentropy')
        test_input = np.random.random((1, 224, 224, 3))
        result = model.predict(test_input)
        
        logger.info(f"Test model prediction shape: {result.shape}")
        logger.info("Test model execution completed - check if it used GPU")
        
        # Check if any ops ran on GPU
        gpu_devices = tf.config.list_logical_devices('GPU')
        logger.info(f"Available GPU devices after test: {gpu_devices}")
        
        return len(gpu_devices) > 0
    except Exception as e:
        logger.error(f"GPU force test failed: {e}")
        return False

# Create Flask app
app = Flask(__name__)

# Create health blueprint
health_bp = Blueprint('health', __name__)

@health_bp.route("/health", methods=["GET"])
def health():
    """Health check endpoint"""
    uptime = time.time() - _start_time
    model_status = {"initialized": _classifier is not None}
    if _initialization_error:
        model_status["error"] = _initialization_error
    
    health_data = {
        "status": "healthy",
        "service": "classifierd",
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
    is_ready = _classifier is not None
    if is_ready:
        return jsonify({
            "ready": True,
            "service": "classifierd"
        })
    else:
        message = "Model not initialized yet"
        if _initialization_error:
            message = f"Model initialization failed: {_initialization_error}"
        return jsonify({
            "ready": False,
            "service": "classifierd",
            "message": message
        }), 503  # Service Unavailable

app.register_blueprint(health_bp)

def get_classifier():
    global _classifier, _initialization_error, gpu_info
    if _classifier is None:
        logger.info("Initializing classifier...")
        try:
            # Force TensorFlow to reinitialize its device detection
            import os
            os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
            os.environ['TF_GPU_THREAD_MODE'] = 'gpu_private'
            os.environ['TF_GPU_THREAD_COUNT'] = '1'
            
            # Import TensorFlow and explicitly initialize CUDA
            import tensorflow as tf
            from speciesnet import SpeciesNet
            
            # More aggressive GPU configuration
            if USE_GPU:
                # Try a different GPU configuration approach
                logger.info("Trying alternative GPU configuration")
                try:
                    # Try rebuilding the GPU devices list
                    physical_devices = tf.config.list_physical_devices('GPU')
                    if not physical_devices:
                        logger.warning("No physical GPUs detected, trying to force device detection")
                        
                        # Try to force GPU detection using environment variables
                        os.environ['CUDA_VISIBLE_DEVICES'] = '0'
                        os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
                        
                        # Reload TensorFlow to apply environment variables
                        import importlib
                        importlib.reload(tf)
                        
                        # Check again
                        physical_devices = tf.config.list_physical_devices('GPU')
                        if physical_devices:
                            logger.info(f"Force-detected GPUs: {physical_devices}")
                    
                    if physical_devices:
                        # Enable memory growth for all GPUs
                        for device in physical_devices:
                            tf.config.experimental.set_memory_growth(device, True)
                        logger.info("Memory growth enabled for all GPUs")
                
                except Exception as e:
                    logger.warning(f"Error in alternative GPU configuration: {e}")
            
            # Use XLA compilation for better performance
            os.environ["TF_XLA_FLAGS"] = "--tf_xla_auto_jit=2"
            
            # Initialize with explicit device control
            logger.info("Initializing SpeciesNet classifier with explicit GPU control")
            _classifier = SpeciesNet(
                model_name="kaggle:google/speciesnet/keras/v4.0.0a", 
                components="classifier", 
                multiprocessing=False
            )
            
            # Check if TensorFlow found a GPU after initialization
            gpu_info["tensorflow"]["available"] = len(tf.config.list_physical_devices('GPU')) > 0
            logger.info(f"Classifier initialized with TensorFlow GPU: {gpu_info['tensorflow']['available']}")
            _initialization_error = None
            
        except Exception as e:
            _initialization_error = str(e)
            logger.warning(f"Error loading classifier: {e}")
            logger.warning("Attempting CPU-only mode")
            
            # Force CPU mode
            old_cuda_visible = os.environ.get('CUDA_VISIBLE_DEVICES', None)
            os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
            
            try:
                # Lazy import
                from speciesnet import SpeciesNet
                _classifier = SpeciesNet(
                    model_name="kaggle:google/speciesnet/keras/v4.0.0a", 
                    components="classifier", 
                    multiprocessing=False
                )
                logger.info("Classifier initialized in CPU-only mode")
                _initialization_error = None
            except Exception as e2:
                _initialization_error = f"Failed in both GPU and CPU modes: {str(e2)}"
                logger.error(f"Failed in CPU mode: {e2}")
                raise
            finally:
                # Restore GPU settings
                if old_cuda_visible is not None:
                    os.environ['CUDA_VISIBLE_DEVICES'] = old_cuda_visible
    return _classifier

# Create shared temp directory
TEMP_DIR = tempfile.mkdtemp(dir=os.getenv("SHARED_TEMP_DIR", "/tmp/shared_temp"))
logger.info(f"Created temp directory: {TEMP_DIR}")

@app.route("/classify", methods=["POST"])
def classify():
    """Endpoint to run species classification on detected regions"""
    # Initialize temp_files list at the beginning of the function
    temp_files = []
    
    try:
        # Get request data
        data = request.get_json()
        if not data:
            abort(400, description="Request body is required")
        
        # Handle both full predictions or just detections
        if "predictions" in data:
            predictions = data["predictions"]
        elif "detections" in data:
            predictions = data["detections"]
        else:
            abort(400, description="Request must contain 'predictions' or 'detections'")
            
        if not isinstance(predictions, list):
            abort(400, description="Predictions must be a list")
            
        # Check if we have images or not
        has_images = False
        for prediction in predictions:
            if prediction.get("image"):
                has_images = True
                break
                
        # Prepare instances
        instances = []
        # temp_files list is now defined at the beginning of the function
        
        if has_images:
            # Process based on base64-encoded images
            for prediction in predictions:
                if "image" not in prediction:
                    abort(400, description="Each prediction must contain an 'image'")
                    
                try:
                    image_data = base64.b64decode(prediction["image"])
                except Exception as e:
                    abort(400, description=f"Invalid base64 data: {str(e)}")
                    
                # Save to temp file
                temp_file = tempfile.NamedTemporaryFile(
                    delete=False, dir=TEMP_DIR, suffix=".jpg"
                )
                temp_files.append(temp_file.name)
                
                with open(temp_file.name, "wb") as f:
                    f.write(image_data)
                    
                # Create instance
                instance = {
                    "filepath": temp_file.name,
                    "detections": prediction.get("detections", []),
                }
                
                # Copy metadata
                for key, value in prediction.items():
                    if key not in ["image", "detections"]:
                        instance[key] = value
                        
                instances.append(instance)
        else:
            # Process based on detections without images
            instances = predictions
            
        # Process with SpeciesNet
        try:
            classifier = get_classifier()
            
            # Create payload
            payload = {"instances": instances}
            
            result = classifier.classify(
                instances_dict=payload,
                run_mode='multi_thread',  # Using multi_thread instead of multi_process
                progress_bars=False
            )
            
            # Remove filepaths from results
            for p in result["predictions"]:
                if "filepath" in p:
                    del p["filepath"]
                    
        except Exception as e:
            abort(500, description=f"Classification error: {str(e)}")
            
        # Clean up
        for temp_file in temp_files:
            try:
                os.remove(temp_file)
            except OSError:
                pass
                
        return jsonify(result)
        
    except Exception as e:
        logger.exception("Error in classify endpoint")
        # Clean up on error
        for temp_file in temp_files:
            try:
                os.remove(temp_file)
            except OSError:
                pass
        abort(500, description=f"Server error: {str(e)}")

# Detect GPUs at startup
detect_gpus()

# Initialize classifier at startup if configured
if INIT_AT_STARTUP:
    try:
        get_classifier()
        logger.info("Pre-initialization complete")
    except Exception as e:
        logger.error(f"Pre-initialization failed: {e}")

if __name__ == "__main__":
    port = int(os.getenv("PORT", 5002))
    logger.info(f"Starting classifier service on port {port}, GPU={USE_GPU}")
    app.run(host="0.0.0.0", port=port, debug=False)