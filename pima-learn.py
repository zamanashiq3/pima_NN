# -*- coding: utf-8 -*-
"""
Created on Mon Nov 21 19:44:57 2016

@author: ashiq
"""


from __future__ import print_function
import numpy as np
np.random.seed(13101002)

filename = 'data.csv'
raw_data = np.loadtxt(filename,delimiter=',')

x_train = raw_data[:,0:8]
y_train = raw_data[:,8]

print("Dataset Shape:")
print("X train:")
print(x_train.shape)
print("Y train:")
print(y_train.shape)


from keras.models import Sequential
from keras.layers import Dense

model = Sequential()
model.add(Dense(12, input_dim=8, init='uniform', activation='relu'))
model.add(Dense(8, init='uniform', activation='relu'))
model.add(Dense(1, init='uniform', activation='sigmoid'))

model.summary()

# Compile model
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
# Fit the model
model.fit(x_train,y_train, nb_epoch=150, batch_size=10)
# evaluate the model
scores = model.evaluate(x_train,y_train)
print("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))
