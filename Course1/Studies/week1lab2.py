import tensorflow as tf
import numpy as np

#Create a 28x28 constant tensor.
orig = tf.constant(np.arange(784), shape =[28,28])

#add a dimension to its first axis.
exp = tf.expand_dims(orig, 0)
print(exp.shape)

#squeeze that dimension out again.
orig_back = tf.squeeze(exp, 0)
print(orig_back.shape == orig.shape)

'''
tf.expand_dims(tensorun_kendisi, eklemek_istedigin_dimension_index)

Not: tf.squueze etmek istedigin dimension mutlaka 1 olmali. Yoksa squeze etmez.
tf.squeeze(tensorun_kendisi, cikarmak_istedigin_dim_index)
'''

flat = tf.reshape(orig, shape = (784,))
print(flat.shape)

'''
Doing Math with the Tensors.
'''

#matrix multiplication.
t1 = tf.constant(np.arange(16), shape=(2,8))
t2 = tf.constant(np.arange(56), shape= (8, 7))

t3 = tf.matmul(t1, t2)
print(t3.shape == (2,7))

'''
Generating random tensors from various distributions.
'''

norm = tf.random.normal(shape=(1920, 1028), mean=0, stddev=1.0, name='random_tensor')
print(norm.shape)
print(tf.reduce_mean(norm))
print(tf.reduce_max(norm))
print(tf.reduce_min(norm))

uniform = tf.random.uniform(shape = (28, 28), maxval=10, minval=1, dtype=tf.float16)
print(uniform)