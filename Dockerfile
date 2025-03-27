FROM ubuntu:22.04 AS builder
WORKDIR /app
RUN apt-get update && apt-get install -y python3.11 python3-pip
COPY requirements.txt .
RUN pip3 install --no-cache-dir -r requirements.txt

FROM ubuntu:22.04
WORKDIR /app

# Install more complete dependencies
RUN apt-get update && apt-get install -y \
    uwsgi \
    uwsgi-plugin-python3 \
    python3.11 \
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

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    SHARED_TEMP_DIR=/tmp/shared_temp \
    LISTEN_PORT=5001

COPY --from=builder /usr/local/lib/python3.11/dist-packages/ /usr/local/lib/python3.11/dist-packages/
COPY . .

# Expose the port
EXPOSE 5001

# Try direct run first for debugging
CMD ["python3", "speciesnetd.py"]

# After it works with direct run, comment above line and 
# uncomment this to use uWSGI in production
# CMD ["uwsgi", "--ini", "uwsgi.ini"]