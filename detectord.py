import os
import time
import logging
import tempfile
import base64
from speciesnet import SpeciesNet

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("detectord")
DEBUG = os.getenv("DEBUG", "false").lower() == "true"
logger.setLevel(logging.DEBUG if DEBUG else logging.INFO)

# Environment config
USE_GPU = os.getenv("USE_GPU", "true").lower() == "true"
MODEL_NAME = "kaggle:google/speciesnet/keras/v4.0.0a"

# Lazy-loaded detector
_detector = None

def get_detector():
    global _detector
    if (_detector is None):
        logger.info("Initializing detector...")
        try:
            _detector = SpeciesNet(
                model_name=MODEL_NAME,
                components="detector",
                multiprocessing=False
            )
            logger.info("Detector initialized")
        except Exception as e:
            logger.error(f"Detector init failed: {e}")
            raise
    return _detector

def handler(event):
    """RunPod Serverless handler accepting a base64 image"""
    start_time = time.time()
    temp_files = []

    try:
        input_data = event.get("input", {})
        if not input_data:
            return {"error": "No input provided"}

        image_data = input_data.get("image")
        if not image_data:
            return {"error": "No 'image' key in input"}

        image_bytes = base64.b64decode(image_data) if isinstance(image_data, str) else image_data
        temp_file = tempfile.NamedTemporaryFile(delete=False, dir="/tmp/shared_temp", suffix=".jpg").name
        with open(temp_file, "wb") as f:
            f.write(image_bytes)
        temp_files.append(temp_file)

        payload = {"instances": [{"filepath": temp_file, "filename": "input.jpg"}]}
        detector = get_detector()
        result = detector.detect(instances_dict=payload, run_mode='multi_thread', progress_bars=False)

        result["predictions"] = [{
            "detections": [d for d in p["detections"] if d["conf"] >= 0.5]
        } for p in result["predictions"]]

        for temp_file in temp_files:
            os.remove(temp_file)

        result["processing_time"] = time.time() - start_time
        return result

    except Exception as e:
        logger.error(f"Handler error: {e}")
        for temp_file in temp_files:
            os.remove(temp_file)
        return {"error": str(e)}

if os.getenv("RUN_LOCAL", "false").lower() == "true":    
    if __name__ == "__main__":
        from flask import Flask, request, jsonify
        import os

        app = Flask(__name__)

        @app.route("/runsync", methods=["POST"])
        def runsync():
            try:
                event = request.json
                os.makedirs("/tmp/shared_temp", exist_ok=True)
                result = handler(event)
                return jsonify(result)
            except Exception as e:
                logger.error(f"Flask endpoint error: {e}")
                return jsonify({"error": str(e)})
                
        @app.route("/health", methods=["GET"])
        def health():
            return jsonify({
                "jobs": {
                    "completed": 4,
                    "failed": 0,
                    "inProgress": 0,
                    "inQueue": 0,
                    "retried": 0
                },
                "workers": {
                    "idle": 1,
                    "initializing": 0,
                    "ready": 1,
                    "running": 0,
                    "throttled": 0,
                    "unhealthy": 0
                }
            })

        # Run a simple test on startup
        with open("test.jpg", "rb") as f:
            image_b64 = base64.b64encode(f.read()).decode()
        test_event = {"input": {"image": image_b64}}
        os.makedirs("/tmp/shared_temp", exist_ok=True)
        result = handler(test_event)
        print("Test result:", result)
        
        # Start the Flask server
        print("Starting Flask server on http://localhost:5001")
        print("Health endpoint available at http://localhost:5001/health")
        app.run(host="0.0.0.0", port=5001)
else:
    # Uncomment for RunPod Serverless
    import runpod
    runpod.serverless.start({"handler": handler})