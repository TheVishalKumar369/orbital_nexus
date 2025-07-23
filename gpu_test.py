import tensorflow as tf
print("Available GPUs:", tf.config.list_physical_devices('GPU'))
tf.debugging.set_log_device_placement(True)

# Try a simple operation on the GPU
with tf.device('/GPU:0'):
    a = tf.constant([[1.0, 2.0, 3.0]])
    b = tf.constant([[4.0, 5.0, 6.0]])
    result = tf.matmul(a, b, transpose_b=True)
    print(result)