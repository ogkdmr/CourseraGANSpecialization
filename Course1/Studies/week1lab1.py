'''
This tutorial is about the Tensorflow functional API.
'''

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.layers import Dense, Input, concatenate
from tensorflow.keras.models import Model
import pandas as pd

#Loading the data. 6 features, binary classfication.
from sklearn.model_selection import train_test_split

df = pd.read_csv("diagnosis.csv")
dataset = df.values

#Split the data into training and test sets.
X_train, X_test, y_train, y_test = train_test_split(dataset[:, :6], dataset[:, 6:], test_size=0.33, random_state=42)

# Assign training and testing inputs/outputs
temp_train, nocc_train, lumbp_train, up_train, mict_train, bis_train = np.transpose(X_train)
temp_test, nocc_test, lumbp_test, up_test, mict_test, bis_test = np.transpose(X_test)

inflam_train, nephr_train = y_train[:, 0], y_train[:, 1]
inflam_test, nephr_test = y_test[:, 0], y_test[:, 1]

#build the input objects.
shape = (1,)

in_temp = Input(shape=shape, name="temp")
in_nocc = Input(shape=shape, name="nocc")
in_lumbp = Input(shape=shape, name="lumbp")
in_up = Input(shape=shape, name="up")
in_mict = Input(shape=shape, name="mict")
in_bis = Input(shape=shape, name="bis")

input_list = [in_temp, in_nocc, in_lumbp, in_up, in_mict, in_bis]
inputs = concatenate(input_list)

#build the output objects.

out_inf = Dense(1, activation = 'sigmoid', name='out_inflam')(inputs)
out_neoph = Dense(1, activation='sigmoid', name='out_neoph')(inputs)

output_list = [out_inf, out_neoph]

#Creating the model.
model = Model(inputs = input_list, outputs=output_list)

'''
#display the model.
tf.keras.utils.plot_model(model, "multi_input_model.png", show_shapes=True, show_dtype=True, show_layer_names=True)
'''

model.compile(loss={'out_inflam': 'binary_crossentropy',
                    'out_neoph': 'binary_crossentropy'},
              metrics = ['acc'],
              loss_weights=[0.5, 0.5],
              optimizer=tf.keras.optimizers.Adam())


history = model.fit([temp_train, nocc_train, lumbp_train, up_train, mict_train, bis_train], [inflam_train, nephr_train], epochs = 1000, batch_size=128,
          validation_data=([temp_test, nocc_test, lumbp_test, up_test, mict_test, bis_test], [inflam_test, nephr_test]))

pd.DataFrame(history.history)[['val_out_inflam_loss', 'val_out_neoph_loss']].plot(figsize=((8,5)))
plt.gca()
plt.show()

model.evaluate([temp_test, nocc_test, lumbp_test, up_test, mict_test, bis_test], [inflam_test, nephr_test], batch_size=32)

