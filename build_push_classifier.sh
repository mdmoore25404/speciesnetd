#!/bin/bash

# Check if version arg is provided
if [ -z "$1" ]; then
    echo "Usage: $0 <version-number>"
    echo "Example: $0 1  # Builds and pushes mdmoore25404/speciesnet-classifier:v1"
    exit 1
fi

# Set version from arg
VERSION="v$1"

# Build the Docker image
echo "Building mdmoore25404/speciesnet-classifier:$VERSION..."
docker build -f Dockerfile.classifier -t mdmoore25404/speciesnet-classifier:"$VERSION" .

# Check if build succeeded
if [ $? -ne 0 ]; then
    echo "Build failed, aborting."
    exit 1
fi

# Push to Docker Hub
echo "Pushing mdmoore25404/speciesnet-classifier:$VERSION..."
docker push mdmoore25404/speciesnet-classifier:"$VERSION"

# Check if push succeeded
if [ $? -ne 0 ]; then
    echo "Push failed."
    exit 1
fi

echo "Done! Tagged as mdmoore25404/speciesnet-classifier:$VERSION"