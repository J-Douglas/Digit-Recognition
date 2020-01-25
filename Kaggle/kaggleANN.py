import os
import pandas as pd
import numpy as np
import keras
from keras.models import Sequential
from keras.layers import *
from keras.utils import to_categorical

train_dataset = pd.read_csv('datasets/train.csv')
test_dataset = pd.read_csv('datasets/test.csv')

train_dataset.head()

train_images = train_dataset.iloc[:, 1:785]
train_labels = train_dataset.iloc[:, 0]

test_images = test_dataset.iloc[:, 0:784]

train_images = train_images.as_matrix().reshape(42000, 784)

test_images = test_images.as_matrix().reshape(28000, 784)

train_images = (train_images/255)-0.7
test_images = (test_images/255)-0.7

# Flattening the images into a 784 dimensional vector
train_images = train_images.reshape((-1, 784))
test_images = test_images.reshape((-1,784))

# Building the model
ANN_model = Sequential()
ANN_model.add(Dense(784, activation='relu', input_dim=784))
ANN_model.add(Dense(300, activation='relu'))
ANN_model.add(Dense(100, activation='relu'))
ANN_model.add(Dense(100, activation='relu'))
ANN_model.add(Dense(100, activation='relu'))
ANN_model.add(Dense(200, activation='relu'))
ANN_model.add(Dense(10, activation='softmax'))

# Compiling the model
ANN_model.compile(
  optimizer='adam', 
  loss='categorical_crossentropy', 
  metrics=['accuracy']
)

# Training the model
epoch_count = 90
ANN_model.fit(
    train_images, 
    to_categorical(train_labels), 
    epochs=epoch_count,
    batch_size=120
)

# Saving the model
ANN_model.save_weights('ANN-models/ANN_Model_{}.h5'.format(epoch_count))

test_pred = pd.DataFrame(ANN_model.predict(test_images, batch_size=60))
test_pred = pd.DataFrame(test_pred.idxmax(axis = 1))
test_pred.index.name = 'ImageId'
test_pred = test_pred.rename(columns = {0: 'Label'}).reset_index()
test_pred['ImageId'] = test_pred['ImageId'] + 1

test_pred.to_csv('submissions/submission_.csv', index = False)

