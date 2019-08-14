# -*- coding: utf-8 -*-
"""
Created on Tue Aug 13 12:55:14 2019

@author: Manish
"""
#Multiclass classification


import tensorflow as tf
import pandas as pd
import numpy as np
# first neural network with keras make predictions
from keras.models import Sequential
from keras.layers import Dense
from sklearn.model_selection import train_test_split


from keras.wrappers.scikit_learn import KerasClassifier
from keras.utils import np_utils
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import LabelEncoder
from sklearn.pipeline import Pipeline
#load data

dataset=pd.read_csv("iris.data", header=None)

# split into input (X) and output (y) variables
x= dataset.iloc[:,0:4].astype(float)
y = dataset.iloc[:,4]


# encode class values as integers
encoder = LabelEncoder()
encoder.fit(y)
y = encoder.transform(y)
y = np_utils.to_categorical(y)
#split
train_x, test_x, train_y, test_y = train_test_split(x, y, train_size=0.5, test_size=0.5, random_state=0)



model = Sequential()

model.add(Dense(16, input_dim=4,activation='relu'))
model.add(Dense(12, activation='relu'))

model.add(Dense(8, activation='relu'))
model.add(Dense(6, activation='relu'))

model.add(Dense(3,activation='softmax'))

# compile the keras model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=["accuracy"])
# fit the keras model on the dataset
model.fit(train_x, train_y, epochs=150, batch_size=10)
# make class predictions with the model
predictions = model.predict_classes(test_x)
print(predictions)

_, accuracy = model.evaluate(test_x, test_y)
print('Accuracy: %.2f' % (accuracy*100))