import tensorflow as tf

'''
Variable stuff

my_var = tf.Variable([1,2,3], dtype=tf.float32)
print(my_var)

my_var.assign([2,4,6])
print(my_var)

print(my_var.shape, my_var.dtype)

my_var.assign_add([1,2,3])
print(my_var)

print(my_var.numpy())
'''


'''
Tensor stuff
'''

from tensorflow.keras.layers import Dense, Input

inputs = Input(shape=(5,))
h = Dense(1, activation="relu")(inputs)
print(h)
print(type(h))

model = tf.keras.models.Model(inputs = inputs, outputs = h)

print(model.inputs)
print(model.outputs)

#Creating a constant tensor.

x = tf.constant([[1,2], [3,4]], dtype=tf.float32)
print(x)

ones = tf.ones_like(x)
print(ones)