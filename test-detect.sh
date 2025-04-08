#!/usr/bin/env bash

# Check for RUNPOD_API_KEY
if [ -z "$RUNPOD_API_KEY" ]; then
    echo "Error: RUNPOD_API_KEY environment variable not set"
    echo "Set it with: export RUNPOD_API_KEY='your-token-here'"
    exit 1
fi

# Convert image to base64, remove newlines
base64 test.jpg | tr -d '\n' > test.b64.oneline

# Create JSON payload in a file
echo "{\"input\":{\"image\":\"$(cat test.b64.oneline)\"}}" > payload.json

# Run curl with auth token from env var
curl -X POST \
  -H "Authorization: Bearer $RUNPOD_API_KEY" \
  -H "Content-Type: application/json" \
  -d @payload.json \
  https://api.runpod.ai/v2/49spki3azmiwt8/runsync

# Clean up
rm test.b64.oneline payload.json