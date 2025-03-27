FROM ubuntu:22.04 
WORKDIR /app

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    SHARED_TEMP_DIR=/tmp/shared_temp \
    LISTEN_PORT=5001 \
    PYTHONDONTWRITEBYTECODE=1

# Install dependencies in a single stage
RUN apt-get update && apt-get install -y \
    python3.11 \
    python3-pip \
    python3-dev \
    python3.11-venv \
    uwsgi \
    uwsgi-plugin-python3 \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev && \
    mkdir -p /var/log/uwsgi && \
    chmod 777 /var/log/uwsgi && \
    mkdir -p /tmp/shared_temp && \
    chmod 777 /tmp/shared_temp && \      
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Make sure pip is properly linked to python3.11
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.11 1 && \
    update-alternatives --set python3 /usr/bin/python3.11 && \
    python3 -m pip install --upgrade pip setuptools wheel

# Copy requirements and install directly in the final image
COPY requirements.txt .
RUN python3.11 -m pip  install --no-cache-dir -r requirements.txt

# Verify installed packages with a more robust check
# RUN python3.11 -c "import sys; print('Python version:', sys.version); import flask; print('Flask version:', flask.__version__); import requests; print('Requests version:', requests.__version__); import speciesnet; print('SpeciesNet module found:', speciesnet.__file__)"

# Copy application code
COPY . .

# Expose the port
EXPOSE 5001

# Use Python 3.11 explicitly
CMD ["python3.11", "speciesnetd.py"]

# For production with uWSGI (uncomment this and comment the line above)
# CMD ["uwsgi", "--ini", "uwsgi.ini", "--plugin", "python3"]