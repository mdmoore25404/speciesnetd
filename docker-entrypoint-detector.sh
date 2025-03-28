#!/bin/bash
set -e

# Print environment info
echo "Starting SpeciesNet Detector Service"
echo "Python: $(python --version)"
echo "GPU enabled: $USE_GPU"
echo "Init at startup: $INIT_AT_STARTUP"
echo "Port: $PORT"

# Create shared temp directory if it doesn't exist
mkdir -p $SHARED_TEMP_DIR
echo "Created shared temp directory: $SHARED_TEMP_DIR"

# Clean up any leftover files from previous runs
find $SHARED_TEMP_DIR -type f -mtime +1 -delete 2>/dev/null || true
echo "Cleaned up old temporary files"

# Set memory optimizations
export MALLOC_ARENA_MAX=2
export PYTHONMALLOC=malloc

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

pip install speciesnet

# Execute the command passed to the script or use the default
if [ "$1" = "flask" ]; then
  echo "Starting Flask development server"
  python detectord.py &
elif [ "$1" = "uwsgi" ] || [ -z "$1" ]; then
  echo "Starting uWSGI server"
  uwsgi --ini detectord.ini &
else
  echo "Running custom command"
  exec "$@" &
fi

child=$!
wait $child