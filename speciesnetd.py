import os
import base64
import tempfile
import shutil
import requests
import json
from flask import Flask, request, jsonify, abort
from werkzeug.utils import secure_filename


from speciesnet import SpeciesNet
my_detector = SpeciesNet(model_name="kaggle:google/speciesnet/keras/v4.0.0a", components="detector", multiprocessing=True)
my_classifier = SpeciesNet(model_name="kaggle:google/speciesnet/keras/v4.0.0a", components="classifier", multiprocessing=True)
my_ensemble = SpeciesNet(model_name="kaggle:google/speciesnet/keras/v4.0.0a", components="ensemble", multiprocessing=True)

app = Flask(__name__)

# Configuration

# Define the tem# Specify the parent directory (must exist and be writable)
# recommend this be an in memory temp dir
SPECIFIC_PATH =  os.getenv("SHARED_TEMP_DIR", "/tmp/shared_temp")

LISTEN_PORT = os.getenv("LISTEN_PORT", 5100) # port that the api listens to


# Ensure the parent directory exists
os.makedirs(SPECIFIC_PATH, exist_ok=True)

# Create a temporary directory under the specific path
TEMP_DIR = tempfile.mkdtemp(dir=SPECIFIC_PATH)
print(f"Created temp directory: {TEMP_DIR}")


def b64_image_to_file(base64_string, filename):
    """Convert a base64 image string to a file."""
    try:
        image_data = base64.b64decode(base64_string)
    except Exception as e:
        raise ValueError(f"Invalid base64 data: {str(e)}")

    with open(filename, "wb") as f:
        f.write(image_data)

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
            
            # print(f"Added instance: {speciesnet_instance}")

            speciesnet_payload["instances"].append(speciesnet_instance)

        # Send request to SpeciesNet API
        try:
                ####speciesnet.detect(instances_dict=instances_dict, run_mode='multi_process', progress_bars=False, predictions_json=None)
                speciesnet_result = my_classifier.ensemble(instances_dict=speciesnet_payload, run_mode='multi_process', progress_bars=False, predictions_json=None)
                ### since filepath is always a tmpfile we can remove it from the speciesnet_result
                for p in speciesnet_result["predictions"]:
                    if "filepath" in p:
                        del p["filepath"]
                    
                # print( json.dumps(speciesnet_result, indent=4) )

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
            
            # print(f"Added instance: {speciesnet_instance}")

            speciesnet_payload["instances"].append(speciesnet_instance)

        # Send request to SpeciesNet API
        try:
                ####speciesnet.detect(instances_dict=instances_dict, run_mode='multi_process', progress_bars=False, predictions_json=None)
                speciesnet_result = my_detector.detect(instances_dict=speciesnet_payload, run_mode='multi_process', progress_bars=False, predictions_json=None)
                # print(speciesnet_result)
                ### since filepath is always a tmpfile we can remove it from the speciesnet_result
                for p in speciesnet_result["predictions"]:
                    if "filepath" in p:
                        del p["filepath"]

                # print( json.dumps(speciesnet_result, indent=4) )
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
        # Clean up files in case of error
        for temp_file in temp_files:
            try:
                os.remove(temp_file)
            except OSError:
                pass
        abort(500, description=f"Server error: {str(e)}")

# @app.teardown_appcontext
# def cleanup_temp_dir(exception=None):
#     """Clean up the temporary directory when the app shuts down."""
#     if os.path.exists(TEMP_DIR):
#         shutil.rmtree(TEMP_DIR, ignore_errors=True)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=LISTEN_PORT, debug=False)