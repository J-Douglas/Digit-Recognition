# Digit Recognition
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

This repository showcases my submissions to the Kaggle Digit Recognizer Competition. The goal of the competition is to create a model which accurately classifies handwritten digits. The competition uses the MNIST ("Modified National Institute of Standards and Technology") dataset to train and test models. Below are two examples of handwritten digits from the MNIST data set. For my first attempt I created an artificial neural network (ANN) which I created in Python using TensorFlow and Keras. More details about the model and my work on this competition are detailed in the sections below.

![Image of 3](https://github.com/J-Douglas/Digit-Recognition/blob/master/mnist/Sample-Testing-Images/3.png)![Image of 8](https://github.com/J-Douglas/Digit-Recognition/blob/master/mnist/Sample-Testing-Images/8.png)

## Kaggle Results (so far)

My best CNN model was 98.9% accurate on Kaggle's testing dataset. It has 3 convolutional layers and 3 pooling layers. It used an epoch of 20 and batch size of 120. Notably, the competition uses a different number of training and testing images than the number that the MNIST database orginally intended; the number of training images was reduced from 60,000 to 42,000 and the number of testing images was increased from 10,000 to 28,000. While the total number of images remains constant, this reorganization of images makes training an accurate model more difficult in the Kaggle competition than using the MNIST database directly.

## Next Steps

I've learned that ANN models have a ceiling of 99% accuracy and to break the 99% barrier a CNN coupled with image augmentation must be used. Here is a link to a post discussing the capabilities of various models in digit recognition: https://www.kaggle.com/c/digit-recognizer/discussion/61480. 

