#!/usr/bin/env python3
"""
Space Debris Tracker - Startup Script
This script helps initialize the project and provides a simple interface.
"""

import os
import sys
import subprocess
from pathlib import Path
import tensorflow as tf

gpus = tf.config.list_physical_devices('GPU')
if gpus:
    print("TensorFlow detected the following GPU(s):")
    for gpu in gpus:
        print(" -", gpu)
else:
    print("No GPU detected by TensorFlow. Using CPU.")

def check_docker():
    """Check if Docker is installed and running"""
    try:
        result = subprocess.run(['docker', '--version'], capture_output=True, text=True)
        if result.returncode == 0:
            print("✅ Docker is installed")
            return True
        else:
            print("❌ Docker is not installed or not running")
            return False
    except FileNotFoundError:
        print("❌ Docker is not installed")
        return False

def check_env_file():
    """Check if .env file exists and has required variables"""
    env_file = Path('.env')
    if not env_file.exists():
        print("❌ .env file not found")
        print("Please copy env.example to .env and add your Space-Track credentials")
        return False
    
    with open(env_file, 'r') as f:
        content = f.read()
        if 'SPACETRACK_USERNAME' in content and 'SPACETRACK_PASSWORD' in content:
            print("✅ .env file configured")
            return True
        else:
            print("❌ .env file missing required credentials")
            return False

def create_directories():
    """Create necessary directories"""
    directories = ['data', 'models', 'templates', 'static']
    for directory in directories:
        Path(directory).mkdir(exist_ok=True)
    print("✅ Directories created")

def run_docker_compose():
    """Run docker-compose commands"""
    print("\n🚀 Starting Space Debris Tracker with Docker...")
    
    # Build the image
    print("Building Docker image...")
    result = subprocess.run(['docker-compose', 'build'], capture_output=True, text=True)
    if result.returncode != 0:
        print("❌ Failed to build Docker image")
        print(result.stderr)
        return False
    
    # Start the services
    print("Starting services...")
    result = subprocess.run(['docker-compose', 'up', '-d'], capture_output=True, text=True)
    if result.returncode != 0:
        print("❌ Failed to start services")
        print(result.stderr)
        return False
    
    print("✅ Services started successfully!")
    print("\n🌐 Access the application at: http://localhost:5000")
    print("📊 View logs with: docker-compose logs -f")
    print("🛑 Stop services with: docker-compose down")
    
    return True

def main():
    """Main startup function"""
    print("🚀 Space Debris Tracker - Startup Script")
    print("=" * 50)
    
    # Check prerequisites
    if not check_docker():
        print("\nPlease install Docker and Docker Compose first:")
        print("https://docs.docker.com/get-docker/")
        return
    
    if not check_env_file():
        print("\nPlease configure your .env file:")
        print("1. Copy env.example to .env")
        print("2. Add your Space-Track.org credentials")
        print("3. Run this script again")
        return
    
    # Create directories
    create_directories()
    
    # Start services
    if run_docker_compose():
        print("\n🎉 FastAPI Space Debris Tracker is ready!")
        print("\nNext steps:")
        print("1. Open http://localhost:5000 in your browser")
        print("2. Check out the API documentation at http://localhost:5000/docs")
        print("3. Click 'Download TLE Data' to fetch debris data")
        print("4. Click 'Parse Data' to process the data")
        print("5. Click 'Train Model' to train the ML model")
        print("6. Click 'Analyze Risks' to assess collision risks")
        print("\n📚 API Documentation:")
        print("   - Swagger UI: http://localhost:5000/docs")
        print("   - ReDoc: http://localhost:5000/redoc")
        print("   - Health Check: http://localhost:5000/health")
    else:
        print("\n❌ Failed to start services. Check the error messages above.")

if __name__ == "__main__":
    main() 