import sys

print(sys.version)
import tensorflow as tf

print(tf.__version__)
import numpy as np

print(np.__version__)
import pandas as pd

print(pd.__version__)

print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices("GPU")))

# Create random tensors
a = tf.random.normal([1000, 1000])
b = tf.random.normal([1000, 1000])

# Perform matrix multiplication on the GPU
c = tf.matmul(a, b)

print("Matrix multiplication result:", c)
print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices("GPU")))
