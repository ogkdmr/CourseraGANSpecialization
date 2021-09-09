import tensorflow as tf
from tqdm import tqdm

phys_devices = tf.config.list_physical_devices()
gpus = tf.config.list_physical_devices('GPU')

#Tensorflow automatically places a tensor on the GPU if there is one available.

x = tf.constant(7.0, shape=(128, 128), name = 'sevens_tensor')
print(x.device) #this will print /job:localhost/replica:0/task:0/device:GPU:0 because I have a GPU.

#Benchmark methods to compare CPU vs GPU.
import time

def time_matadd(x, device):
    start = time.time()
    for _ in range(10):
        tf.add(x, x)
    time_lapsed = time.time() - start
    print('10x matrix addition took: {} seconds on {}'.format(1000 * time_lapsed, device))
    return time_lapsed

def time_matmul(x, device):
    start = time.time()
    for _ in tqdm(range(10000)):
        tf.matmul(x,x)
    time_lapse = time.time() - start
    print('10 matmul ops took {} secs on {}'.format(time_lapse, device))
    return time_lapse

#Benchmarking on the CPU.

cpu_add = 0; cpu_matmul = 0 ; gpu_add = 0; gpu_matmul = 0

with tf.device('CPU:0'): # this is how you force tensors into a device.
    x = tf.random.uniform((1000, 1000))
    assert x.device.endswith('CPU:0')
    cpu_add = time_matadd(x, 'CPU')
    cpu_matmul = time_matmul(x, 'CPU')

#Benchmarking on the GPU.

with tf.device('GPU:0'):
    x = tf.random.uniform((1000, 1000))
    assert x.device.endswith('GPU:0')
    gpu_add = time_matadd(x, 'GPU')
    gpu_matmul = time_matmul(x, 'GPU')

#Print the GPU acceleration coefficient.
print('GPU acceleration on addition is: {}x'.format(cpu_add / gpu_add))
print('GPU acceleration on matmul is: {}x'.format(cpu_matmul / gpu_matmul))

'''
Result: Substantial GPU acceleration is observed in both operations.
GPU acceleration on addition is: 11.800632911392405x
GPU acceleration on matmul is: 7.381904251038376x
'''