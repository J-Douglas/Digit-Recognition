# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import keras
from keras.models import Sequential
from keras.layers import *
from keras.utils import to_categorical
from keras.datasets import mnist

# Any results you write to the current directory are saved as output.

(train_images, train_labels), (test_images, test_labels) = mnist.load_data()
train_images = train_images.reshape((60000, 28, 28, 1))
train_images = train_images.astype(float32) / 255
test_images = test_images.reshape((10000, 28, 28, 1))
test_images = test_images.astype(‘float32’) / 255
train_labels = to_categorical(train_labels)
test_labels = to_categorical(test_labels)

# Building the model
CNN_model = Sequential()
CNN_model.add(layers.Conv2D(32,(5,5),activation=’relu’,input_shape=(28,28,1)))
CNN_model.add(layers.MaxPooling2D((2, 2)))
CNN_model.add(layers.Conv2D(64, (5, 5), activation=’relu’))
CNN_model.add(layers.MaxPooling2D((2, 2)))
CNN_model.add(Dense(10, activation='softmax'))

# Compiling the model
CNN_model.compile(
  optimizer='adam', 
  loss='categorical_crossentropy', 
  metrics=['accuracy']
)

# Training the model
epoch_count = 90
CNN_model.fit(
    train_images, 
    to_categorical(train_labels), 
    epochs=epoch_count,
    batch_size=120
)
