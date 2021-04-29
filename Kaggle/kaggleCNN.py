# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import tensorflow as tf
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import keras
from keras.models import Sequential
from keras.layers import *
from keras.utils import to_categorical
from keras.datasets import mnist
from keras.preprocessing.image import ImageDataGenerator

# Any results you write to the current directory are saved as output.

### Load directly from MNIST dataset
# (train_images, train_labels), (test_images, test_labels) = mnist.load_data()

## Loading dataframe from Kaggle dataset
train_dataset = pd.read_csv('datasets/train.csv')
test_dataset = pd.read_csv('datasets/test.csv')

train_dataset.head()

train_images = train_dataset.iloc[:, 1:785]
train_labels = train_dataset.iloc[:, 0]

test_images = test_dataset.iloc[:, 0:784]

train_images = train_images.to_numpy().reshape((42000, 28, 28, 1))

test_images = test_images.to_numpy().reshape((28000, 28, 28, 1))

### Data augmentation to move images around
# datagen = ImageDataGenerator(
# 	rotation_range=15,
# 	width_shift_range=0.1,
# 	height_shift_range=0.1)

train_labels = to_categorical(train_labels)

# Building the model
CNN_model = Sequential()
CNN_model.add(Conv2D(32,(5,5), activation='relu', input_shape=(28,28,1)))
CNN_model.add(MaxPooling2D((2, 2)))
CNN_model.add(Conv2D(64, (5, 5), activation='relu'))
CNN_model.add(MaxPooling2D((2, 2)))
CNN_model.add(Conv2D(128, (3, 3), activation='relu'))
CNN_model.add(MaxPooling2D((2, 2)))
CNN_model.add(Flatten())
CNN_model.add(Dense(10, activation='softmax'))

# Compiling the model
CNN_model.compile(
  optimizer='adam', 
  loss='categorical_crossentropy', 
  metrics=['accuracy']
)

CNN_model.summary()

# print("\nTraining images numpy shape: {}\n".format(train_images.shape))
# print("Training labels numpy shape: {}\n".format(train_labels.shape))

epoch_count = 20
batch_count = 120

# Data Augmentation Version 1
# for e in range(epoch_count):
# 	print("Epoch: {}".format(e))
# 	batches = 0
# 	for x_batch, y_batch in datagen.flow(train_images,train_labels, batch_size=50):
# 			CNN_model.fit(x_batch,y_batch)
# 			batches += 1
# 			if batches >= len(train_images)//100:
# 				break

### Data Augmentation Version 2
def augment_data(dataset, dataset_labels, augementation_factor=1, use_random_rotation=True, use_random_shear=True, use_random_shift=True, use_random_zoom=True):
	augmented_image = []
	augmented_image_labels = []

	for num in range (0, dataset.shape[0]):

		for i in range(0, augementation_factor):
			# original image:
			augmented_image.append(dataset[num])
			augmented_image_labels.append(dataset_labels[num])

			if use_random_rotation:
				augmented_image.append(tf.keras.preprocessing.image.random_rotation(dataset[num], 20, row_axis=0, col_axis=1, channel_axis=2))
				augmented_image_labels.append(dataset_labels[num])

			if use_random_shear:
				augmented_image.append(tf.keras.preprocessing.image.random_shear(dataset[num], 0.2, row_axis=0, col_axis=1, channel_axis=2))
				augmented_image_labels.append(dataset_labels[num])

			if use_random_shift:
				augmented_image.append(tf.keras.preprocessing.image.random_shift(dataset[num], 0.2, 0.2, row_axis=0, col_axis=1, channel_axis=2))
				augmented_image_labels.append(dataset_labels[num])

			if use_random_zoom:
				augmented_image.append(tf.keras.preprocessing.image.random_zoom(dataset[num], (0.9,0.9), row_axis=0, col_axis=1, channel_axis=2))
				augmented_image_labels.append(dataset_labels[num])

	return np.array(augmented_image), np.array(augmented_image_labels)

augmented_train_images, augmented_train_labels = augment_data(train_images, train_labels)

# Training with regular dataset
CNN_model.fit(
	augmented_train_images,
	augmented_train_labels,
	epochs=epoch_count,
	batch_size=batch_count)

# Saving the model
CNN_model.save_weights('CNN-models/CNN_Model_{}.h5'.format(epoch_count))

test_pred = pd.DataFrame(CNN_model.predict(test_images, batch_size=batch_count))
test_pred = pd.DataFrame(test_pred.idxmax(axis = 1))
test_pred.index.name = 'ImageId'
test_pred = test_pred.rename(columns = {0: 'Label'}).reset_index()
test_pred['ImageId'] = test_pred['ImageId'] + 1

test_pred.to_csv('submissions/submission_5.csv', index = False)