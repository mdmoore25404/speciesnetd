"""
Common utilities for both detector and classifier services.
This module contains shared code to avoid duplication.
"""
import os
import sys
import base64
import tempfile
import logging
import time
import json
from flask import Flask, request, jsonify, abort, Blueprint
import multiprocessing

# Set up logging
def setup_logging(name):
    logging.basicConfig(level=logging.INFO, 
                       format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    return logging.getLogger(name)

def configure_gpu():
    """Configure GPU settings via environment variables"""
    # TensorFlow configuration for cleaner operation
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"           # Reduce TensorFlow logging
    os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"   # Prevent TF from grabbing all GPU memory
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"     # Match CUDA device IDs to hardware order
    os.environ["TF_USE_CUDNN_AUTOTUNE"] = "0"          # Disable cuDNN autotuning

def setup_multiprocessing(logger, use_gpu=False):
    """Set up multiprocessing with the appropriate start method for CUDA"""
    if use_gpu and multiprocessing.current_process().name == 'MainProcess':
        try:
            if hasattr(multiprocessing, 'set_start_method'):
                multiprocessing.set_start_method('spawn', force=True)
                logger.info("Set multiprocessing start method to 'spawn' for CUDA compatibility")
        except RuntimeError as e:
            logger.warning(f"Could not set multiprocessing start method: {e}")

def create_health_blueprint(service_name, start_time, model=None, initialization_error=None, gpu_info=None):
    """Create a Blueprint with health and readiness endpoints"""
    bp = Blueprint('health', __name__)

    @bp.route("/health", methods=["GET"])
    def health():
        """Health check endpoint for monitoring and container orchestration."""
        uptime = time.time() - start_time
        
        # Check if model has been initialized
        model_status = {"initialized": model is not None}
        if initialization_error:
            model_status["error"] = initialization_error
        
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
            "service": service_name,
            "uptime_seconds": round(uptime, 2),
            "gpu_info": gpu_info or {},
            "gpu_memory": gpu_memory_info,
            "model_status": model_status,
            "python_version": sys.version,
            "environment": {
                "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES", "not set"),
                "TF_FORCE_GPU_ALLOW_GROWTH": os.environ.get("TF_FORCE_GPU_ALLOW_GROWTH", "not set")
            }
        }
        
        # Add version info if available
        try:
            import speciesnet
            health_data["speciesnet_info"] = {
                "module_path": speciesnet.__file__,
                "version": getattr(speciesnet, "__version__", "unknown")
            }
            
            if service_name == "detectord":
                try:
                    import torch
                    health_data["torch_version"] = torch.__version__
                except ImportError:
                    pass
            else:  # classifier
                try:
                    import tensorflow as tf
                    health_data["tensorflow_version"] = tf.__version__
                except ImportError:
                    pass
        except (ImportError, AttributeError):
            health_data["speciesnet_info"] = None
            
        return jsonify(health_data)

    @bp.route("/ready", methods=["GET"])
    def ready():
        """Readiness check endpoint that verifies if the model is loaded."""
        is_ready = model is not None
        
        if is_ready:
            return jsonify({
                "ready": True,
                "service": service_name,
                "model_loaded": True
            })
        else:
            return jsonify({
                "ready": False,
                "service": service_name,
                "message": f"Model for {service_name} has not been initialized yet",
                "error": initialization_error
            }), 503  # Service Unavailable
            
    return bp