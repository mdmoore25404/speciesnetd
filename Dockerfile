FROM ubuntu:22.04 AS builder
WORKDIR /app

# Install build dependencies
RUN apt-get update && apt-get install -y \
    python3.11 \
    python3-pip \
    python3.11-venv \
    python3-dev && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Setup virtual environment
RUN python3.11 -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip setuptools wheel && \
    pip install --no-cache-dir -r requirements.txt

# Final image
FROM ubuntu:22.04
WORKDIR /app

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    SHARED_TEMP_DIR=/tmp/shared_temp \
    LISTEN_PORT=5001 \
    PYTHONDONTWRITEBYTECODE=1 \
    PATH="/opt/venv/bin:$PATH"

# Copy virtual environment from builder
COPY --from=builder /opt/venv /opt/venv

# Install runtime dependencies
RUN apt-get update && apt-get install -y \
    python3.11 \
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

# Copy application code
COPY . .

# Expose the port
EXPOSE 5001

# Use Python from virtual environment
CMD ["python", "speciesnetd.py"]

# For production with uWSGI (uncomment this and comment the line above)
# CMD ["uwsgi", "--ini", "uwsgi.ini", "--plugin", "python3", "--virtualenv", "/opt/venv"]