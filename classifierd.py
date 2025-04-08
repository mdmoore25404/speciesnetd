import os
import time
import logging
import tempfile
import base64
from speciesnet import SpeciesNet

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("classifierd")
DEBUG = os.getenv("DEBUG", "false").lower() == "true"
logger.setLevel(logging.DEBUG if DEBUG else logging.INFO)

USE_GPU = os.getenv("USE_GPU", "true").lower() == "true"
MODEL_NAME = "kaggle:google/speciesnet/keras/v4.0.0a"

_classifier = None

def get_classifier():
    global _classifier
    if _classifier is None:
        logger.info("Initializing classifier...")
        try:
            if not USE_GPU:
                os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
            _classifier = SpeciesNet(
                model_name=MODEL_NAME,
                components="classifier",
                multiprocessing=False
            )
            logger.info("Classifier initialized")
        except Exception as e:
            logger.error(f"Classifier init failed: {e}")
            raise
    return _classifier

def handler(event):
    start_time = time.time()
    temp_files = []

    try:
        # Skip dummy test input
        if event.get("id") == "local_test":
            logger.debug("Skipping local test input")
            return {"status": "skipped", "message": "Waiting for real job"}

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
        classifier = get_classifier()
        result = classifier.classify(
            instances_dict=payload,
            run_mode='multi_thread',
            progress_bars=False
        )

        logger.debug(f"Raw classifier result: {result}")
        if "predictions" in result:
            if isinstance(result["predictions"], list) and all("detections" in p for p in result["predictions"]):
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
        logger.info("Running local test...")
        with open("test.jpg", "rb") as f:
            image_b64 = base64.b64encode(f.read()).decode()
        test_event = {"input": {"image": image_b64}}
        os.makedirs("/tmp/shared_temp", exist_ok=True)
        result = handler(test_event)
        print(result)
else:
    import runpod
    logger.info("Starting RunPod Serverless...")
    runpod.serverless.start({"handler": handler})