#!/bin/bash
set -e

# Print environment info
echo "Starting SpeciesNet Detector Service"
echo "Python: $(python --version)"
echo "GPU enabled: $USE_GPU"
echo "Init at startup: $INIT_AT_STARTUP"
echo "Port: $PORT"
echo "Running in RunPod environment: $RUNPOD_ENV"

# Ensure hostname is properly set for multiprocessing
export HOSTNAME=$(hostname)
echo "Running with hostname: $HOSTNAME"

# For RunPod, ensure the shared memory is usable
# This is equivalent to --shm-size in Docker
if [ -d "/dev/shm" ]; then
  echo "Shared memory status:"
  df -h /dev/shm || echo "Failed to check shared memory"
  # Try to increase shared memory if possible
  mount -o remount,size=2G /dev/shm 2>/dev/null || echo "Failed to increase shared memory (this is expected in RunPod)"
fi

# Create shared temp directory if it doesn't exist
mkdir -p $SHARED_TEMP_DIR
echo "Created shared temp directory: $SHARED_TEMP_DIR"
chmod 1777 $SHARED_TEMP_DIR

# Clean up any leftover files from previous runs
find $SHARED_TEMP_DIR -type f -mtime +1 -delete 2>/dev/null || true
echo "Cleaned up old temporary files"

# Check network connectivity
echo "Network configuration:"
ip addr || echo "ip command not available"

# Set memory optimizations
export MALLOC_ARENA_MAX=2
export PYTHONMALLOC=malloc

# Disable multiprocessing for containers
export OMP_NUM_THREADS=1  # OpenMP threads
export MULTIPROCESSING_SEMAPHORES=0
export SPECIESNET_DISABLE_MULTIPROCESSING=true  # Custom env var we'll check in our code

# Check available memory
free -m || echo "free command not available"
df -h || echo "df command not available"

# Handle termination signals properly
_term() {
  echo "Received SIGTERM, shutting down..."
  kill -TERM "$child" 2>/dev/null
  wait "$child"
  echo "Shutdown complete"
  exit 0
}

trap _term SIGTERM SIGINT

# Initialize the model in the entrypoint script to avoid Flask/gunicorn worker issues
if [ "$INIT_AT_STARTUP" = "true" ]; then
  echo "Pre-initializing detector model..."
  # Run a simple Python script to initialize the model
  python -c "
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'  # Start with CPU for safety
from speciesnet import SpeciesNet
print('Importing SpeciesNet...')
try:
    detector = SpeciesNet(model_name='kaggle:google/speciesnet/keras/v4.0.0a', components='detector', multiprocessing=False)
    print('Successfully initialized detector in CPU mode')
except Exception as e:
    print(f'Error initializing detector: {e}')
"
  echo "Pre-initialization complete"
fi

# Execute the command passed to the script or use the default
if [ "$1" = "flask" ]; then
  echo "Starting Flask development server"
  python detectord.py &
elif [ "$1" = "gunicorn" ] || [ -z "$1" ]; then
  echo "Starting gunicorn server"
  gunicorn --bind 0.0.0.0:$PORT --timeout 300 --workers 1 detectord:app &
elif [ "$1" = "uwsgi" ]; then
  echo "Starting uWSGI server"
  uwsgi --ini detectord.ini &
else
  echo "Running custom command"
  exec "$@" &
fi

child=$!
wait $child