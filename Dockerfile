FROM ubuntu:22.04

LABEL org.opencontainers.image.source="https://github.com/mdmoore25404/tcam/speciesnetd"

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    SHARED_TEMP_DIR=/tmp/shared_temp \
    LISTEN_PORT=5001 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1

# Create necessary directories first (combined into one RUN to reduce layers)
RUN mkdir -p /var/log/uwsgi ${SHARED_TEMP_DIR} && \
    chmod 777 /var/log/uwsgi ${SHARED_TEMP_DIR}

WORKDIR /app

# Install system dependencies (combining apt operations reduces layers)
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    python3.11 \
    python3-pip \
    curl \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Copy only requirements first to leverage Docker cache
COPY ./requirements.txt .

# Install Python dependencies
RUN pip3 install --no-cache-dir -r requirements.txt uwsgi

# Copy application code (isolate this change to minimize cache invalidation)
COPY . .

# GPU test as the final step before running
RUN python3 -m speciesnet.scripts.gpu_test

EXPOSE 5001

# Run uWSGI with the ini file
CMD ["uwsgi", "--ini", "uwsgi.ini"]