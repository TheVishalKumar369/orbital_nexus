#!/usr/bin/env python3
"""
GPU Configuration Module for TensorFlow
This module provides functions to configure TensorFlow for optimal GPU usage
"""

import tensorflow as tf
import os
import logging

def configure_gpu_memory():
    """
    Configure TensorFlow GPU memory growth and allocation
    This prevents TensorFlow from allocating all GPU memory at startup
    """
    try:
        # Get list of physical GPU devices
        gpus = tf.config.experimental.list_physical_devices('GPU')
        
        if gpus:
            # Set memory growth for each GPU
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            
            # Optional: Set memory limit (uncomment if needed)
            # memory_limit = 3072  # MB (for 4GB GPU, leave some for system)
            # tf.config.experimental.set_memory_limit(gpus[0], memory_limit)
            
            logging.info(f"GPU memory growth configured for {len(gpus)} GPU(s)")
            return True
        else:
            logging.warning("No GPUs found, using CPU")
            return False
            
    except Exception as e:
        logging.error(f"Error configuring GPU memory: {e}")
        return False

def set_mixed_precision():
    """
    Enable mixed precision training for better performance on modern GPUs
    """
    try:
        policy = tf.keras.mixed_precision.Policy('mixed_float16')
        tf.keras.mixed_precision.set_global_policy(policy)
        logging.info("Mixed precision policy set to mixed_float16")
        return True
    except Exception as e:
        logging.error(f"Error setting mixed precision: {e}")
        return False

def configure_tensorflow_for_production():
    """
    Configure TensorFlow for optimal production performance
    """
    # Set logging level
    tf.get_logger().setLevel('INFO')
    
    # Configure GPU
    gpu_configured = configure_gpu_memory()
    
    # Set mixed precision for better performance (optional)
    # mixed_precision_set = set_mixed_precision()
    
    # Print configuration summary
    print("TensorFlow Configuration Summary:")
    print(f"- GPU Available: {len(tf.config.experimental.list_physical_devices('GPU')) > 0}")
    print(f"- GPU Configured: {gpu_configured}")
    print(f"- TensorFlow Version: {tf.__version__}")
    print(f"- CUDA Available: {tf.test.is_built_with_cuda()}")
    
    return gpu_configured

def get_device_info():
    """
    Get detailed information about available devices
    """
    devices = {
        'cpu': tf.config.experimental.list_physical_devices('CPU'),
        'gpu': tf.config.experimental.list_physical_devices('GPU')
    }
    
    info = {
        'cpu_count': len(devices['cpu']),
        'gpu_count': len(devices['gpu']),
        'cuda_available': tf.test.is_built_with_cuda(),
        'tensorflow_version': tf.__version__
    }
    
    return info

# Environment variable checks
def check_gpu_environment():
    """
    Check GPU-related environment variables
    """
    gpu_env_vars = {
        'CUDA_VISIBLE_DEVICES': os.environ.get('CUDA_VISIBLE_DEVICES', 'Not set'),
        'NVIDIA_VISIBLE_DEVICES': os.environ.get('NVIDIA_VISIBLE_DEVICES', 'Not set'),
        'TF_FORCE_GPU_ALLOW_GROWTH': os.environ.get('TF_FORCE_GPU_ALLOW_GROWTH', 'Not set'),
        'TF_GPU_MEMORY_FRACTION': os.environ.get('TF_GPU_MEMORY_FRACTION', 'Not set')
    }
    
    return gpu_env_vars

if __name__ == "__main__":
    # Test the configuration
    configure_tensorflow_for_production()
    
    print("\nDevice Information:")
    info = get_device_info()
    for key, value in info.items():
        print(f"- {key}: {value}")
    
    print("\nEnvironment Variables:")
    env_vars = check_gpu_environment()
    for key, value in env_vars.items():
        print(f"- {key}: {value}")
