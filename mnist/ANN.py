import os
import numpy as np 
import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D
from keras.utils import to_categorical
import random
import matplotlib.pyplot as plt 

# Training and testing images/labels
train_images = mnist.train_images() 
train_labels = mnist.train_labels() 
test_images = mnist.test_images() 

test_labels = mnist.test_labels()   

# Normalizing the images
train_images = (train_images/255)-0.7
test_images = (test_images/255)-0.7

# Flattening the images into a 784 dimensional vector
train_images = train_images.reshape((-1, 784))
test_images = test_images.reshape((-1,784))

# Building the model
ANN_model = Sequential()
ANN_model.add(Dense(784, activation='relu', input_dim=784))
ANN_model.add(Dense(64, activation='relu'))
ANN_model.add(Dense(64, activation='relu'))
ANN_model.add(Dense(10, activation='softmax'))

# Compiling the model
ANN_model.compile(
	optimizer='adam', 
	loss='categorical_crossentropy', 
	metrics=['accuracy']
)

# Training the model
epoch_count = 20
ANN_model.fit(
    train_images, 
    to_categorical(train_labels), 
    epochs=epoch_count,
    batch_size=80
)

# Saving the model
ANN_model.save_weights('ANN-Models/ANN_Model_{}.h5'.format(epoch_count))

# Predictions vs. testing images
num_predictions = 5
random_lower_bound = random.randint(0,test_images.shape[0]-num_predictions)
random_upper_bound = random_lower_bound + num_predictions
predictions = ANN_model.predict(test_images[random_lower_bound:random_upper_bound])
print (np.argmax(predictions,axis =1))
print(test_labels[random_lower_bound:random_upper_bound])

# Plotting tests
for i in range(random_lower_bound,random_upper_bound):
  first_image = test_images[i]
  first_image = np.array(first_image, dtype='float')
  pixels = first_image.reshape((28, 28))
  plt.imshow(pixels, cmap='gray')
  plt.show()
  plt.savefig('Sample-Testing-Images/{}.png'.format(test_labels[i]))

