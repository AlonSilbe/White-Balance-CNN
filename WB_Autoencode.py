

import numpy as np
import matplotlib.pyplot as plt
import os
import cv2 
import random 
import tensorflow

from keras.models import Model
import tensorflow as tf
from tensorflow import keras
from keras.models import Model
from keras.layers import Conv2D, MaxPooling2D, Dense, Input, Conv2D, UpSampling2D, BatchNormalization

from keras.callbacks import EarlyStopping
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D, BatchNormalization, UpSampling2D
datez="180922_input_8TH_run_input3_500ephochs"
#Size of image in the network. Each image has 300x300 pixels
IMG_SIZE=300

#ARRAYs with the corect images and images with bad color
wrong_color=[]
good_color=[]

#counters
p=0
p1=0

#path=r"C:\Users\Alon1000\Documents\End Project Azrieli\set02\input02"
path=r"C:\Users\Alon1000\Documents\End Project Azrieli\set02\input2"
for img in os.listdir(path):  # iterate over each image in folder
        imgz = cv2.imread(os.path.join(path,img) ,cv2.IMREAD_COLOR)
        new_array = cv2.resize(imgz, (IMG_SIZE, IMG_SIZE))  # resize to normalize data size
        wrong_color.append(new_array)  # add this to our training_data
        print("Saved distorted Color picture number:",p)
        p+=1

#path=r"C:\Users\Alon1000\Documents\End Project Azrieli\set02\truth02"
path=r"C:\Users\Alon1000\Documents\End Project Azrieli\set02\truth2"
for img in os.listdir(path):  # iterate over each image in folder
        imgz = cv2.imread(os.path.join(path,img) ,cv2.IMREAD_COLOR)
        new_array = cv2.resize(imgz, (IMG_SIZE, IMG_SIZE))  # resize to normalize data size
        good_color.append(new_array)  # add this to our training_data
        print("Saved corect Color picture number",p1)
        p1+=1

wrong_color=(np.array(wrong_color))/255
good_color=(np.array(good_color))/255
from sklearn.model_selection import train_test_split
Wrong_color_train, Wrong_colot_test, good_color_train, good_color_test = train_test_split(wrong_color, good_color, test_size=0.1)

def create_model():
  x = tensorflow.keras.Input(shape=(300, 300, 3))
# Encoder
  e_conv1 = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
  pool1 = MaxPooling2D((2, 2), padding='same')(e_conv1)
  batchnorm_1 = BatchNormalization()(pool1)
  e_conv2 = Conv2D(32, (3, 3), activation='relu', padding='same')(batchnorm_1)
  pool2 = MaxPooling2D((2, 2), padding='same')(e_conv2)
  batchnorm_2 = BatchNormalization()(pool2)
  e_conv3 = Conv2D(16, (3, 3), activation='relu', padding='same')(batchnorm_2)
  h = MaxPooling2D((2, 2), padding='same')(e_conv3)
# Decoder
  d_conv1 = Conv2D(64, (3, 3), activation='relu', padding='same')(h)
  up1 = UpSampling2D((2, 2))(d_conv1)
  d_conv2 = Conv2D(32, (3, 3), activation='relu', padding='same')(up1)
  up2 = UpSampling2D((2, 2))(d_conv2)
  d_conv3 = Conv2D(16, (3, 3), activation='relu')(up2)
  up3 = UpSampling2D((2, 2))(d_conv3)
  r = Conv2D(3, (3, 3), activation='sigmoid', padding='same')(up3)
  model = Model(x, r)
  model.compile(optimizer='Adam', loss='mse')
  return model
WB_CNN = create_model()
#WB_CNN =keras.models.load_model("saved_model_7run_150922_input_7TH_run_input3_500ephochs_for_real.h5")
WB_CNN_early_stop = EarlyStopping(monitor='loss', patience=3)
WB_CNN_History = WB_CNN.fit(Wrong_color_train, good_color_train, epochs=500, batch_size=32, validation_data=(Wrong_colot_test, good_color_test))





result = WB_CNN.predict(Wrong_colot_test)
WB_CNN.evaluate(Wrong_colot_test, good_color_test)
WB_CNN.save("saved_model_8run_{}.h5".format(datez))


#plot_rgb_img(add_gaussian_nt.plot(WB_CNN_History.epoch, WB_CNN_History.history['loss'])
plt.plot(WB_CNN_History.epoch, WB_CNN_History.history['loss'])
plt.title('Epochs on Training Loss 8th run')
plt.xlabel('# of Epochs')
plt.ylabel('Mean Squared Error')
#plt.savefig("Graph_{}_{}.jpg".format(datez,random.randint(0,99)))
plt.show()


plt.title("Results of CNN on 5th run of the program")
plt.subplot(1,3,1)
plt.title("Original")
plt.imshow(good_color_test[0])
ax = plt.gca()

  #hide x-axis
ax.get_xaxis().set_visible(False)

  #hide y-axis 
ax.get_yaxis().set_visible(False)
plt.subplot(1,3,2)
plt.title("Bad color")
plt.imshow(Wrong_colot_test[0])
ax = plt.gca()

  #hide x-axis
ax.get_xaxis().set_visible(False)

  #hide y-axis 
ax.get_yaxis().set_visible(False)
plt.subplot(1,3,3)
plt.title("Reconstructed")
plt.imshow(result[0])
ax = plt.gca()

  #hide x-axis
ax.get_xaxis().set_visible(False)

  #hide y-axis 
ax.get_yaxis().set_visible(False)
plt.show()
cv2.imwrite("saved_imgs/resultz_{}_{}.png".format(datez,random.randint(0,45)),result[0])

