#!/usr/bin/env python3
"""
GPU Test Script for TensorFlow
This script tests if TensorFlow can detect and use GPU properly
"""

import tensorflow as tf
import os

def test_gpu_setup():
    print("=" * 50)
    print("TensorFlow GPU Setup Test")
    print("=" * 50)
    
    # Check TensorFlow version
    print(f"TensorFlow version: {tf.__version__}")
    
    # Check if GPU is available
    print(f"GPU Available: {tf.config.experimental.list_physical_devices('GPU')}")
    
    # List all physical devices
    physical_devices = tf.config.experimental.list_physical_devices()
    print(f"All Physical Devices: {physical_devices}")
    
    # Check CUDA build
    print(f"Built with CUDA: {tf.test.is_built_with_cuda()}")
    
    # Check GPU devices
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        print(f"Number of GPUs: {len(gpus)}")
        for i, gpu in enumerate(gpus):
            print(f"GPU {i}: {gpu}")
            
        # Try to configure GPU memory growth
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            print("GPU memory growth configured successfully")
        except Exception as e:
            print(f"Error configuring GPU memory growth: {e}")
    else:
        print("No GPUs found!")
    
    # Test basic GPU computation
    print("\n" + "=" * 30)
    print("Testing GPU Computation")
    print("=" * 30)
    
    try:
        with tf.device('/GPU:0'):
            # Create random matrices
            a = tf.random.normal([1000, 1000])
            b = tf.random.normal([1000, 1000])
            
            # Perform matrix multiplication
            import time
            start_time = time.time()
            c = tf.matmul(a, b)
            gpu_time = time.time() - start_time
            
            print(f"GPU computation completed in {gpu_time:.4f} seconds")
            print(f"Result shape: {c.shape}")
            print("GPU computation test: PASSED")
            
    except Exception as e:
        print(f"GPU computation test failed: {e}")
        
        # Fallback to CPU
        print("Trying CPU computation...")
        try:
            with tf.device('/CPU:0'):
                start_time = time.time()
                c = tf.matmul(a, b)
                cpu_time = time.time() - start_time
                print(f"CPU computation completed in {cpu_time:.4f} seconds")
                print("Falling back to CPU")
        except Exception as e2:
            print(f"CPU computation also failed: {e2}")
    
    # Environment variables
    print("\n" + "=" * 30)
    print("Environment Variables")
    print("=" * 30)
    cuda_vars = ['CUDA_VISIBLE_DEVICES', 'NVIDIA_VISIBLE_DEVICES', 
                 'TF_FORCE_GPU_ALLOW_GROWTH', 'TF_GPU_MEMORY_FRACTION']
    for var in cuda_vars:
        value = os.environ.get(var, 'Not set')
        print(f"{var}: {value}")

if __name__ == "__main__":
    test_gpu_setup()
