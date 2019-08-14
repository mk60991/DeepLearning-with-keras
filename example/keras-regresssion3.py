# -*- coding: utf-8 -*-
"""
Created on Wed Aug 14 11:51:36 2019

@author: Manish
"""

#regression using keras
import pandas as pd
import numpy as np
# Import required libraries
import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt
import sklearn

# Import necessary modules
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from math import sqrt

# Keras specific
import keras
from keras.models import Sequential
from keras.layers import Dense
# Read dataset into X and Y
df = pd.read_csv(r'C:\Users\Manish\Desktop\housing.csv', delim_whitespace=True, header=None)


X = df.iloc[:, 0:13].values
Y = df.iloc[:, 13].values

#print "X: ", X
#print "Y: ", Y
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.30, random_state=40)
print(X_train.shape)
print(X_test.shape)



# Define model
model = Sequential()
model.add(Dense(500, input_dim=13, activation= "relu"))
model.add(Dense(100, activation= "relu"))
model.add(Dense(50, activation= "relu"))

model.add(Dense(1))


model.compile(loss= "mean_squared_error" , optimizer="adam", metrics=["mean_squared_error"])
model.fit(X_train, y_train, epochs=20)


pred= model.predict(X_test)
print(pred)

arr=np.array([5.20177,0,18.1,1,0.77,6.127,83.4,2.7227,24,666,20.3,395.43,11.48]).reshape(1,-1)
predd= model.predict(arr)
print(predd)

results = model.evaluate(X_test, y_test)
print(model.metrics_names)     # list of metric names the model is employing
print(results)                 # actual figure of metrics computed


print('loss: ', results[0])
print('mse: ', results[1])