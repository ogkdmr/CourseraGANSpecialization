'''
Transfer learning the VGG16 model.
'''

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

#avoiding memory issues.
physical_devices = tf.config.list_physical_devices('GPU')
for device in physical_devices:
    tf.config.experimental.set_memory_growth(device, True)


#importing the vgg model.
from tensorflow.keras.applications import VGG19
vgg_model = VGG19()

vgg_input = vgg_model.input
vgg_layers = vgg_model.layers

#building a new model that outputs features from the layers of the vgg19.
feature_model = tf.keras.models.Model(inputs = vgg_input, outputs = [layer.output for layer in vgg_layers])
print(feature_model.summary())

#see the extracted features from a random image.
random_img = np.random.randn(1, 224, 224, 3).astype('float32')
features = feature_model(random_img)
print(np.argmax(features[-1].numpy()))
#interesting note: this model classifies Gaussian noise as 'mosquito net' with 6% confidence. Which makes sense, actually.

#now let's explore a real image, let's display it here. it's a showel.
img = Image.open('hard.jpeg')
plt.imshow(img)
plt.show()

#remember these modules.
from tensorflow.keras.applications.vgg19 import preprocess_input
from tensorflow.keras.preprocessing import image

#load and preprocess the image.
img = image.load_img('hard.jpeg', target_size=(224, 224))
img = image.img_to_array(img)
img = np.expand_dims(img, axis=0) #adding a first dimension as the vgg input requires.
img = preprocess_input(img) #this function comes with the model itself.

#display the preprocessed version for VGG19. It looks interesting.
plt.imshow(img.squeeze(0))
plt.show()

#extract the features and see the class.
img_feats = feature_model(img)
clf = np.argmax(img_feats[-1]) #792:shovel

#now let's visualize some intermediate feature maps.
feature_map = img_feats[10]
plt.imshow(feature_map[:,:,:,0].numpy().squeeze(0))
plt.show()

#extract output from a layer given its name.
block1_pool_model = tf.keras.models.Model(feature_model.input, outputs=feature_model.get_layer('block1_pool').output)
feats = block1_pool_model(img)

fmaps = feats[0,:,:]
plt.figure(figsize=(15,15))
for n in range(3): # for 3 channels (RGB).
    ax = plt.subplot(1,3,n+1)
    plt.imshow(fmaps[:,:,n])
    plt.axis('off')
plt.subplots_adjust(wspace=0.01, hspace=0.01)
plt.show()


