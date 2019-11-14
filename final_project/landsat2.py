# -*- coding: utf-8 -*-
"""
Created on Sat Oct 27 12:26:13 2018

@author: mgreen13
"""

import tensorflow as tf
import numpy as np
import pandas as pd
from keras.callbacks import TensorBoard
from keras.layers import Conv2D, MaxPool2D, Dense, Dropout, Flatten
from keras.models import Sequential

x_train = pd.read_csv(tf.gfile.Open('X_train_sat6.csv'),header=None) #This will kill the available memory on the kaggle machine :\
y_train = pd.read_csv(tf.gfile.Open('y_train_sat6.csv'),header=None)
x_train = x_train.values.reshape(x_train.shape[0],28,28,4).astype(np.float32)
y_train = y_train.values.astype(np.float32)

x_eval = pd.read_csv(tf.gfile.Open('X_test_sat6.csv'),header=None)
y_eval = pd.read_csv(tf.gfile.Open('y_test_sat6.csv'),header=None)
x_eval = x_eval.values.reshape(x_eval.shape[0],28,28,4).astype(np.float32)
y_eval = y_eval.values.astype(np.float32)

print(x_train.shape)
print(y_train.shape)
print(x_eval.shape)
print(y_eval.shape)

xTrainSmall=x_train[0:100,:,:,:]
yTrainSmall=y_train[0:100,:]
xEvalSmall=x_eval[0:100,:,:,:]
yEvalSmall=y_eval[0:100,:]




#create model
tf.reset_default_graph()

model = Sequential()
model.add(Conv2D(16, (3,3), activation='relu', input_shape=(28,28,4)))
model.add(Conv2D(32, (3,3), activation='relu'))
model.add(MaxPool2D(pool_size=(2,2)))
model.add(Dropout(0.5))
model.add(Conv2D(32, (3,3), activation='relu'))
model.add(Conv2D(64, (3,3), activation='relu'))
model.add(MaxPool2D(pool_size=(2,2)))
model.add(Dropout(0.5))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(6, activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

print(model.summary())

tbcallback = TensorBoard(log_dir='./Graph/', histogram_freq=1, write_graph=True, write_grads=True)

model.fit(xTrainSmall, yTrainSmall, batch_size=200, epochs=10, verbose=1, validation_data=(xEvalSmall, yEvalSmall), callbacks=[tbcallback])
