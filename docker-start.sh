#!/bin/bash

# Docker startup script for Space Debris Tracking System

echo "Starting Space Debris Tracking System with Docker..."

# Check if Docker is running
if ! docker info > /dev/null 2>&1; then
    echo "Error: Docker is not running. Please start Docker first."
    exit 1
fi

# Check if nvidia-docker runtime is available (for GPU support)
if docker info 2>/dev/null | grep -q nvidia; then
    echo "NVIDIA Docker runtime detected - GPU support will be available"
    COMPOSE_FILE="docker-compose.yml"
else
    echo "Warning: NVIDIA Docker runtime not detected - using CPU-only mode"
    COMPOSE_FILE="docker-compose.cpu.yml"
fi

# Build and start the containers
echo "Building and starting containers..."
docker-compose -f $COMPOSE_FILE up --build -d

# Wait for services to be ready
echo "Waiting for services to start..."
sleep 10

# Check service health
echo "Checking service health..."
docker-compose -f $COMPOSE_FILE ps

# Show logs for a few seconds
echo "Recent logs:"
docker-compose -f $COMPOSE_FILE logs --tail=20

echo ""
echo "Space Debris Tracking System is now running!"
echo "Access the application at: http://localhost:5000"
echo "Redis is available at: localhost:6379"
echo ""
echo "To view logs: docker-compose -f $COMPOSE_FILE logs -f"
echo "To stop: docker-compose -f $COMPOSE_FILE down"
