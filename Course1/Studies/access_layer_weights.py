import tensorflow as tf
from tensorflow.keras.layers import Dense, Conv1D, Flatten, Input, AveragePooling1D

inputs = Input(shape=(128, 1), name='weather_input')
h = Conv1D(16, 5, activation='relu', name='conv1')(inputs)
h = AveragePooling1D(3, name='average_pooling_layer')(h)
h = Flatten(name='flatten_layer')(h)
outputs = Dense(20, activation='sigmoid', name='dense_layer')(h)

model = tf.keras.models.Model(inputs, outputs)

'''

These functions are used for getting the tensorflow variables for the layers and their weights. 
We can explore these, but we cannot pass these to other networks.

#Print the layers of the model.
print(model.layers)

#Print the weights of the last layer as tensorflow variables.
print(model.layers[-1].weights)

#Kernel and bias printed separately.
print(model.layers[-1].kernel)
print(model.layers[-1].bias)

#print weights as numpy array.
print(model.layers[-1].get_weights())

#access the layer by its name using get_layer()
print(model.get_layer('dense_layer').get_weights()[0].shape) # this will print the kernel of the dense layer.

'''


'''
This part explores extracting tensors out of the layers of a network.
These tensors can be pushed to other networks for transfer learning. 
'''

#get the input and outputs of layers and the model.

print(model.input, model.output)
print(model.get_layer('dense_layer').input.shape)
print(model.get_layer('dense_layer').output.shape)


#Build a new model using the weights from this model.
flatten_output = model.get_layer('flatten_layer').output
model2 = tf.keras.models.Model(inputs=model.input, outputs=flatten_output)

#Use an entire model as part of another model (Sequential way)

model3 = tf.keras.models.Sequential([
    model2,
    Dense(20, activation='relu', name='final_dense_layer')
])

#print((model3.summary()))

#Use an entire model as part of another model (Functional way)

dense_output = Dense(20, activation='relu', name='final_dense_layer')(model2.output)
new_model = tf.keras.models.Model(inputs=model.input, outputs=dense_output)
print(new_model.summary())
