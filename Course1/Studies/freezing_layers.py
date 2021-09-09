'''
GIST: We can freeze a layer as follows:
model.get_layer('layer_name').trainable = False
This way when you are training the model, parameters of this layer does not change.
Used for transfer learning.
'''

from tensorflow.keras.models import Sequential
from tensorflow.keras import layers
import matplotlib.pyplot as plt
import numpy as np

model = Sequential([
    layers.Dense(4, input_shape=(4,), activation='relu', kernel_initializer='random_uniform', bias_initializer='ones'),
    layers.Dense(2, activation='relu', kernel_initializer='lecun_normal', bias_initializer='ones'),
    layers.Dense(4, activation='softmax')
])

model.summary()

#get the weights and biases of the layers.
def get_weights(model):
    return [e.weights[0].numpy() for e in model.layers]

def get_biases(model):
    return [e.weights[1].numpy() for e in model.layers]

#W0, b0 are pre-training weights and biases. W1, b1 are post-training weights and biases.
def plot_delta_weights(W0_layers, W1_layers, b0_layers, b1_layers):
    plt.figure(figsize=(8,8))
    for n in range(3):
        delta_l = W1_layers[n] - W0_layers[n]
        print('Layer' + str(n) + 'bias variation: ', np.linalg.norm(b1_layers[n]) - b0_layers[n])
        ax = plt.subplot(1,3, n+1)
        plt.imshow(delta_l)
        plt.title('Layer{}'.format(n))
        plt.axis('off')
    plt.colorbar()
    plt.suptitle('Weight matrices variation')
    plt.show()

#Construct random dummy input.
X_train, X_test = np.random.random((100, 4)), np.random.random((20, 4))
y_train, y_test = X_train, X_test

W0_layers = get_weights(model)
b0_layers = get_biases(model)

#compile and train the model
model.compile(loss='mse', optimizer='sgd', metrics=['mse'])
model.fit(X_train, y_train, epochs=100, batch_size=10, validation_data=(X_test, y_test))

W1_layers = get_weights(model)
b1_layers = get_biases(model)

plot_delta_weights(W0_layers, W1_layers, b0_layers, b1_layers)

'''
Now run the code above by freezing the entire model before training. 
Add model.trainable = False before compilation and see that after training none
of the weights change.
'''

#Note: This gives us the trainable parameters of a model.
trainable_params = model.trainable_variables
print(type(trainable_params), len(trainable_params))
print()
print(trainable_params)