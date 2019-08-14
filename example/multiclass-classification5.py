# -*- coding: utf-8 -*-
"""
Created on Wed Aug 14 16:20:36 2019

@author: Manish
"""
#multiclass classification on mnist dataset
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from keras.datasets import mnist
from keras.models import Sequential
from keras.utils.np_utils import to_categorical
from keras.models import Sequential
from keras.layers import Activation, Dense
from keras import optimizers

(X_train, y_train), (X_test, y_test) = mnist.load_data()

"""
# reshaping X data: (n, 28, 28) => (n, 784)

X_train = X_train.reshape(60000, 784)
X_test = X_test.reshape(10000, 784)
X_train = X_train.astype("float32")
X_test = X_test.astype("float32")

# Put everything on grayscale
X_train /= 255
X_test /= 255

# convert class vectors to binary class matrices
Y_train = np_utils.to_categorical(y_train, 10)
Y_test = np_utils.to_categorical(y_test, 10)
"""
# reshaping X data: (n, 28, 28) => (n, 784)
X_train = X_train.reshape((X_train.shape[0], X_train.shape[1] * X_train.shape[2]))
X_test = X_test.reshape((X_test.shape[0], X_test.shape[1] * X_test.shape[2]))

# converting y data into categorical (one-hot encoding)
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

# use only 33% of training data to expedite the training process
train_x, test_x, train_y, test_y = train_test_split(X_train, y_train, test_size = 0.67, random_state = 7)

model = Sequential()
model.add(Dense(50, input_dim=784,activation='relu'))

model.add(Dense(50, activation='relu'))
model.add(Dense(50, activation='relu'))

model.add(Dense(50, activation='relu'))
model.add(Dense(10,activation='softmax'))


# compile the keras model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=["accuracy"])
# fit the keras model on the dataset
model.fit(train_x, train_y, epochs=150, batch_size=10)
# make class predictions with the model
predictions = model.predict_classes(test_x)
print(predictions)

_, accuracy = model.evaluate(test_x, test_y)
print('Accuracy: %.2f' % (accuracy*100))