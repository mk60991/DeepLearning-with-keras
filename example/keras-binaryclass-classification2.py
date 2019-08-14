# -*- coding: utf-8 -*-
"""
Created on Tue Aug 13 10:15:52 2019

@author: Manish
"""


#binary class classification
import tensorflow as tf
import pandas as pd
import numpy as np
# first neural network with keras make predictions
from numpy import loadtxt
from keras.models import Sequential
from keras.layers import Dense
from sklearn.model_selection import train_test_split


#load data
url="https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv"
dataset=pd.read_csv(url, names=["preg","glu","bp","st","insulin","bmi","diabetes","age","otcome"])


# split into input (X) and output (y) variables
X = dataset.iloc[:,0:8]
y = dataset.iloc[:,8]

#split
train_X, test_X, train_y, test_y = train_test_split(X, y, train_size=0.5, test_size=0.5, random_state=0)


# define the keras model
model = Sequential()
model.add(Dense(12, input_dim=8, activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(1, activation='sigmoid'))
# compile the keras model
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
# fit the keras model on the dataset
model.fit(train_X, train_y, epochs=150, batch_size=10)
# make class predictions with the model
predictions = model.predict_classes(test_X)
print(predictions)


# evaluate the keras model
_, accuracy = model.evaluate(X, y)
print('Accuracy: %.2f' % (accuracy*100))






