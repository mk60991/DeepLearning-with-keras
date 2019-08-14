import numpy as np #Algebra
import pandas as pd #data precessing, csv I/O

from keras.utils import to_categorical
from sklearn.model_selection import train_test_split

data_train=pd.read_csv("C:\\Users\\soumyama\\Documents\\Python Scripts\\personal tutorial\\kaggle\\Fashion\\fashion-mnist_train.csv")
data_test=pd.read_csv("C:\\Users\\soumyama\\Documents\\Python Scripts\\personal tutorial\\kaggle\\Fashion\\fashion-mnist_test.csv") 

img_rows=28
img_cols=28
input_shape=(img_rows, img_cols, 1)

X=np.array(data_train.iloc[:,1:])
y= to_categorical(np.array(data_train.iloc[:, 0]))

#Split teh training data into train and validation data
X_train, X_valid, y_train,  y_valid= train_test_split(X, y, test_size=0.2, random_state=42)

#Test data
X_test=np.array(data_test.iloc[:,1:])
y_test=to_categorical(np.array(data_test.iloc[:,0]))


X_train= X_train.reshape(X_train.shape[0], img_rows, img_cols, 1)
X_test= X_test.reshape(X_test.shape[0], img_rows, img_cols, 1)
X_valid= X_valid.reshape(X_valid.shape[0], img_rows, img_cols, 1)

 
X_train=X_train.astype('float32')
X_test=X_test.astype('float32')
X_valid=X_valid.astype('float32')
X_train/=255
X_test/=255
X_valid/=255


import keras
from keras.models import Sequential
from keras .layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout
from keras.layers.normalization import BatchNormalization

batch_size=256
num_classes=10
epochs=50

#input image dimensions

img_rows, img_cols=28, 28


model=Sequential()
model.add(Conv2D(32, kernel_size=(3,3), activation='relu', kernel_initializer='he_normal',
                 input_shape=input_shape))
model.add(MaxPooling2D((2,2)))
model.add(Dropout(0.25))
model.add(Conv2D(64,(3,3), activation='relu'))
model.add(MaxPooling2D((2,2)))
model.add(Dropout(0.25))
model.add(Conv2D(128,(3,3), activation='relu'))
model.add(Dropout(0.4))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(num_classes, activation='softmax'))

model.compile(loss=keras.losses.categorical_crossentropy, optimizer= keras.optimizers.Adam(),
              metrics=['accuracy'])


model.summary()


#Training
history=model.fit(X_train, y_train, batch_size=batch_size, epochs=epochs, verbose=1, 
                  validation_data=(X_valid, y_valid))
 
score=model.evaluate(X_test, y_test, verbose=0)

print("test loss: ",score[0])
print("test accuracy: ",score[1])

#Result: Loss - 0.195 Accuracy - 92.64%


#Plot training and validation accuracy

import matplotlib.pyplot as plt
%matplotlib inline

accuracy=history.history['acc']
valid_accuracy=history.history['val_acc']
loss=history.history['loss']
valid_loss=history.history['val_loss']
epochs= range(len(accuracy))

#Plot for Accuracy
plt.plot(epochs, accuracy, 'bo', label='Training Accuracy')
plt.plot(epochs, valid_accuracy, 'b', label='Validation Accuracy')
plt.title("Training and Validation Accuracy")
plt.legend()
plt.figure()
plt.show()

#Plot for Loss
plt.plot(epochs, loss, 'bo', label='Training Loss')
plt.plot(epochs, valid_loss, 'b', label='Validation Loss')
plt.title("Training and Validation Loss")
plt.legend()
plt.figure()
plt.show()


#Classification Report
#Get the predictions for the test data
predicted_classes=model.predict_classes(X_test)
predicted_classes

#Get the indices to be plotted
y_true = data_test.iloc[:, 0]
correct = np.nonzero(predicted_classes==y_true)[0]
incorrect = np.nonzero(predicted_classes!=y_true)[0]

from sklearn.metrics import classification_report
target_names=["Class {}".format(i) for i in range(num_classes)]
print(classification_report(y_true, predicted_classes, target_names=target_names))


#Visualise correct and incorrect precisions
for i, correct in enumerate(correct[:9]):
    plt.subplot(3,3,i+1)
    plt.imshow(X_test[correct].reshape(28,28), cmap='gray', interpolation='none')
    plt.title("Predicted {}, Class {}".format(predicted_classes[correct], y_true[correct]))
    plt.tight_layout()

for i, incorrect in enumerate(incorrect[0:9]):
    plt.subplot(3,3,i+1)
    plt.imshow(X_test[incorrect].reshape(28,28), cmap='gray', interpolation='none')
    plt.title("Predicted {}, Class {}".format(predicted_classes[incorrect], y_true[incorrect]))
    plt.tight_layout()



#What do activations look like
test_im=X_train[154]
plt.imshow(test_im.reshape(28,28), cmap='viridis', interpolation='none')
plt.show()
 

#Lets see the activation of the 2nd channel of 1st layer of CNN
from keras import models
layer_outputs=[layer.output for layer in model.layers[:8]]
activation_model=models.Model(input=model.input, output=layer_outputs)
activations=activation_model.predict(test_im.reshape(1,28,28,1))

first_layer_activation=activations[0]
plt.matshow(first_layer_activation[0,:,:,4], cmap='viridis')


#Plot teh activations for other conv layers as well
layer_names=[]
for layer in model.layers[:-1]:
    layer_names.append(layer.name)
images_per_row=16
for layer_name, layer_activation in zip(layer_names, activations):
    if layer_name.startswith('conv'):
        n_features=layer_activation.shape[-1]
        size=layer_activation.shape[1]
        n_col=n_features//images_per_row
        display_grid=np.zeros((size*n_col, images_per_row*size))
        for col in range(n_col):
            for row in range(image_per_row):
                channel_image=layer_activation[0,:,:,col*images_per_row+row]
                channel_image-=channel_image.mean()
                channel_image/=channel_image.std()
                channel_image*=64
                channel_image+=128
                channel_image=np.clip(channel_image, 0,255).astype('uint8')
                display_grid[col*size :(col+1)*size,
                             row*size:(row+1)*size]=channel_image
        scale=1./size
        plt.figure(figsize=(scale*display_grid.shape[1], scale*display_grid.shape[0]))
        plt.title("Layer Name")
        plt.grid(False)
        plt.imshow(display_grid, aspect='auto', cmap='viridis')


   