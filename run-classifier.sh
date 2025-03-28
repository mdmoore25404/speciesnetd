#!/bin/bash
set -e

echo "Starting SpeciesNet Service with GPU=${USE_GPU}"

# Create shared directory
mkdir -p /tmp/shared_temp
chmod 1777 /tmp/shared_temp

# Install SpeciesNet if needed - pip will skip if already installed
pip install --no-cache-dir speciesnet || echo "Warning: Could not install speciesnet"

# Run the application
exec "$@"