#!/bin/bash

# Podman startup script for Space Debris Tracking System

echo "Starting Space Debris Tracking System with Podman..."

# Check if Podman is installed
if ! command -v podman &> /dev/null; then
    echo "Error: Podman is not installed. Please install Podman first."
    exit 1
fi

# Check if podman-compose is installed
if ! command -v podman-compose &> /dev/null; then
    echo "Error: podman-compose is not installed. Please install podman-compose first."
    echo "Install with: pip install podman-compose"
    exit 1
fi

# Check for GPU support
if command -v nvidia-smi &> /dev/null; then
    echo "NVIDIA GPU detected - GPU support will be available"
    COMPOSE_FILE="podman-compose.yml"
else
    echo "Warning: NVIDIA GPU not detected - using CPU-only mode"
    COMPOSE_FILE="podman-compose.cpu.yml"
fi

# Build and start the containers
echo "Building and starting containers..."
podman-compose -f $COMPOSE_FILE up --build -d

# Wait for services to be ready
echo "Waiting for services to start..."
sleep 10

# Check service health
echo "Checking service health..."
podman-compose -f $COMPOSE_FILE ps

# Show logs for a few seconds
echo "Recent logs:"
podman-compose -f $COMPOSE_FILE logs --tail=20

echo ""
echo "Space Debris Tracking System is now running!"
echo "Access the application at: http://localhost:5000"
echo "Redis is available at: localhost:6379"
echo ""
echo "To view logs: podman-compose -f $COMPOSE_FILE logs -f"
echo "To stop: podman-compose -f $COMPOSE_FILE down"
