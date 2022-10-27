import numpy as np
import matplotlib.pyplot as plt
import os
import cv2 
import random 
import tensorflow
import tensorflow as tf
from tensorflow import keras
from keras.models import Model
from keras.layers import Conv2D, MaxPooling2D, Dense, Input, Conv2D, UpSampling2D, BatchNormalization

from keras.callbacks import EarlyStopping
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D, BatchNormalization, UpSampling2D
from grayworld import white_balance


IMG_SIZE=300
wrong_color=[]
good_color=[]

#Load images and resize them to fit our model

path=r"bad_img.jpg"
imgz = cv2.imread(path ,cv2.IMREAD_COLOR)
new_array = cv2.resize(imgz, (IMG_SIZE, IMG_SIZE))  # resize to normalize data size
wrong_color.append(new_array)  # add this to our training_data

path=r"img.jpg"        
imgz2 = cv2.imread(path ,cv2.IMREAD_COLOR)
new_array2 = cv2.resize(imgz2, (IMG_SIZE, IMG_SIZE))  # resize to normalize data size
good_color.append(new_array2)  # add this to our training_data
        

wrong_color=(np.array(wrong_color))/255
good_color=(np.array(good_color))/255


WB_Autoencoder = keras.models.load_model("model.h5")
#gaussian_early_stop = EarlyStopping(monitor='loss', patience=3)
#gaussian_history = gaussian_auto_encoder.fit(Wrong_color_train, good_color_train, epochs=50, batch_size=32, validation_data=(Wrong_colot_test, good_color_test), callbacks=[gaussian_early_stop])
result = WB_Autoencoder.predict(wrong_color)
WB_Autoencoder.evaluate(wrong_color, good_color)




plt.subplot(1,3,1)
plt.title("Original")
	#get current axes
ax = plt.gca()

	#hide x-axis
ax.get_xaxis().set_visible(False)

	#hide y-axis 
ax.get_yaxis().set_visible(False)
plt.imshow(new_array2)

plt.subplot(1,3,2)
plt.title("With color distortion")
ax = plt.gca()

	#hide x-axis
ax.get_xaxis().set_visible(False)

	#hide y-axis 
ax.get_yaxis().set_visible(False)
plt.imshow(new_array)

plt.subplot(1,3,3)
plt.title("White Balance")
plt.imshow(result[0])
ax = plt.gca()

	#hide x-axis
ax.get_xaxis().set_visible(False)

	#hide y-axis 
ax.get_yaxis().set_visible(False)
plt.show()





plt.subplot(1,4,1)
plt.title("Original")
	#get current axes
ax = plt.gca()

	#hide x-axis
ax.get_xaxis().set_visible(False)

	#hide y-axis 
ax.get_yaxis().set_visible(False)
plt.imshow(new_array2)

plt.subplot(1,4,2)
plt.title("Bad Image")
ax = plt.gca()

	#hide x-axis
ax.get_xaxis().set_visible(False)

	#hide y-axis 
ax.get_yaxis().set_visible(False)
plt.imshow(new_array)

plt.subplot(1,4,3)
plt.title("Our W.B")
plt.imshow(result[0])
ax = plt.gca()

	#hide x-axis
ax.get_xaxis().set_visible(False)

	#hide y-axis 
ax.get_yaxis().set_visible(False)


plt.subplot(1,4,4)
plt.title("Gray world method ")
plt.imshow(white_balance(new_array2))
ax = plt.gca()

	#hide x-axis
ax.get_xaxis().set_visible(False)

	#hide y-axis 
ax.get_yaxis().set_visible(False)
plt.show()

