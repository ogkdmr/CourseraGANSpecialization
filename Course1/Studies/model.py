from tensorflow.keras.layers import Input, Dense, Conv2D, AveragePooling2D, Flatten, BatchNormalization, Concatenate
from tensorflow.keras.models import Model
import numpy as np
import tensorflow as tf

physical_devices = tf.config.experimental.list_physical_devices('GPU')
assert len(physical_devices) > 0, "Not enough GPU hardware devices available"
config = tf.config.experimental.set_memory_growth(physical_devices[0], True)

#adding an additional input and output.

input_image = Input(shape=(128, 128, 1,), name="image")
input_boxcoords = Input(shape=(4,), name="box_coords")

h = Conv2D(16, 5, activation='relu')(input_image)
h = AveragePooling2D(4)(h)
h = BatchNormalization()(h)
h = Flatten()(h)
h = Concatenate()([h, input_boxcoords])
outputs = Dense(10, activation="softmax", name="class")(h)
aux_output = Dense(4, activation='linear', name="box")(h)

model = Model(inputs=[input_image, input_boxcoords], outputs=[outputs, aux_output])
print(model.summary())

model.compile(
    loss = {'class': "categorical_crossentropy", 'box': 'mse'},
    optimizer='adam',
    loss_weights={'class': 0.6, 'box':0.4},
    metrics={"class":'accuracy', 'box':'mse'}
)

X_images = np.random.randn(4096, 128, 128, 1)
X_boxes = np.random.randn(4096, 4)


y_images = np.random.randint(low=0, high=10, size=4096, dtype='int')
y_boxes = np.random.randn(4096, 4)
y_images = tf.keras.utils.to_categorical(y_images, num_classes=10)



model.fit({'image': X_images, 'box_coords': X_boxes},
          {'class': y_images, 'box':y_boxes},
          batch_size=64, epochs=20,
          validation_split=0.2)
