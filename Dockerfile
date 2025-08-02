# Use the official TensorFlow GPU image as a parent image
FROM tensorflow/tensorflow:2.13.0-gpu

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Configure GPU memory growth
ENV TF_FORCE_GPU_ALLOW_GROWTH=true
ENV TF_CPP_MIN_LOG_LEVEL=1
ENV CUDA_VISIBLE_DEVICES=0

# Set python3 and pip3 to point to Python 3.9
# RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.9 1

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# (TensorFlow is already installed in the base image)

COPY . .

# Create necessary directories
RUN mkdir -p data models notebooks scripts

# Set environment variables
ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1

# Expose port
EXPOSE 5000

# Run the application with uvicorn
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "5000"] 