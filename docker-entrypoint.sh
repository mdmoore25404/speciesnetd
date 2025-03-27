#!/usr/bin/env bash

cd /app

pip install "speciesnet>=4.0.3,<4.1.0"

# Check if USE_UWSGI environment variable is set and not false
if [ "${USE_UWSGI:-false}" != "false" ]; then
    echo "Starting with uWSGI..."
    exec uwsgi --ini uwsgi.ini --plugin python3 --virtualenv /opt/venv
else
    echo "Starting with Python directly..."
    exec python speciesnetd.py
fi
