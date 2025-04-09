#!/usr/bin/env bash

# Check for RUN_LOCAL environment variable
if [ "${RUN_LOCAL,,}" = "true" ]; then
    API_URL="http://localhost:5002/runsync"
    echo "Running against local endpoint: $API_URL"
    HEADERS=("-H" "Content-Type: application/json")
else
    # Check for RUNPOD_API_KEY
    if [ -z "$RUNPOD_API_KEY" ]; then
        echo "Error: RUNPOD_API_KEY environment variable not set"
        echo "Set it with: export RUNPOD_API_KEY='your-token-here'"
        exit 1
    fi
    API_URL="https://api.runpod.ai/v2/n5rymetyso1ruu/runsync"
    echo "Running against RunPod endpoint: $API_URL"
    HEADERS=("-H" "Authorization: Bearer $RUNPOD_API_KEY" "-H" "Content-Type: application/json")
fi

# Convert image to base64, remove newlines
base64 test.jpg | tr -d '\n' > test.b64.oneline

# Create JSON payload in a file
echo "{\"input\":{\"image\":\"$(cat test.b64.oneline)\"}}" > payload.json

# Run curl with appropriate headers and URL
curl -X POST \
  "${HEADERS[@]}" \
  -d @payload.json \
  $API_URL

# Clean up
rm test.b64.oneline payload.json