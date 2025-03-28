import tensorflow as tf
import os

# Set environment variables first
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "0"  # Show all logs
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"

print(f"TensorFlow version: {tf.__version__}")
print(f"Is GPU available: {tf.test.is_gpu_available()}")
print(f"Is built with CUDA: {tf.test.is_built_with_cuda()}")
print(f"Physical devices: {tf.config.list_physical_devices()}")
print(f"GPU devices: {tf.config.list_physical_devices('GPU')}")

# Try to run a simple operation on GPU
if len(tf.config.list_physical_devices('GPU')) > 0:
    print("Attempting GPU computation...")
    with tf.device('/GPU:0'):
        a = tf.constant([[1.0, 2.0], [3.0, 4.0]])
        b = tf.constant([[1.0, 1.0], [1.0, 1.0]])
        c = tf.matmul(a, b)
        print(f"GPU computation result: {c}")
else:
    print("No GPU devices detected by TensorFlow")

# Check CUDA and cuDNN versions if possible
try:
    print(f"CUDA version: {tf.sysconfig.get_build_info()['cuda_version']}")
    print(f"cuDNN version: {tf.sysconfig.get_build_info()['cudnn_version']}")
except:
    print("Could not get CUDA/cuDNN version info")