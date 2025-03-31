#!/bin/bash
set -e

echo "Starting SpeciesNet Service with GPU=${USE_GPU}"

# Create shared directory
mkdir -p /tmp/shared_temp
chmod 1777 /tmp/shared_temp

# Set up virtual environment in /workspace for speciesnet
VENV_DIR="/workspace/venv"
if [ ! -d "$VENV_DIR" ]; then
    echo "Creating virtual environment in $VENV_DIR..."
    python -m venv "$VENV_DIR"
    source "$VENV_DIR/bin/activate"
    pip install --no-cache-dir speciesnet || echo "Warning: Could not install speciesnet"
else
    echo "Using existing virtual environment from $VENV_DIR..."
    source "$VENV_DIR/bin/activate"
fi

# Run the application
exec "$@"