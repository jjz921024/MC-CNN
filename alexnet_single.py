#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar 31 16:03:15 2018

@author: gdutllc
"""
from keras.models import Sequential
from keras.layers import Dense,Flatten,Dropout
from keras.layers.convolutional import Conv2D,MaxPooling2D
#from keras.utils.np_utils import to_categorical

class Alex:
    def build(self):   #input
        self.model = Sequential()
        #self.model.add(input)
        self.model.add(Conv2D(96,(11,11),strides=(4,4),input_shape=(224,224,3),
                              padding='valid',activation='relu',
                              kernel_initializer='uniform'))
        self.model.add(MaxPooling2D(pool_size=(3,3),strides=(2,2)))
        
        self.model.add(Conv2D(256,(5,5),strides=(1,1),
                              padding='same',activation='relu',kernel_initializer='uniform'))
        self.model.add(MaxPooling2D(pool_size=(3,3),strides=(2,2)))
        
        self.model.add(Conv2D(384,(3,3),strides=(1,1),padding='same',
                              activation='relu',kernel_initializer='uniform'))
        self.model.add(Conv2D(384,(3,3),strides=(1,1),padding='same',
                              activation='relu',kernel_initializer='uniform'))
        self.model.add(Conv2D(256,(3,3),strides=(1,1),padding='same',
                              activation='relu',kernel_initializer='uniform'))
        self.model.add(MaxPooling2D(pool_size=(3,3),strides=(2,2)))
        
        self.model.add(Flatten())
        self.model.add(Dense(4096,activation='relu'))
        self.model.add(Dropout(0.5))
        self.model.add(Dense(4096,activation='relu'))
        self.model.add(Dropout(0.5))
        self.model.add(Dense(49,activation='softmax'))
        self.model.compile(loss='categorical_crossentropy',optimizer='sgd',metrics=['accuracy'])
        return self.model
        
        
        
    
#%%

from keras.callbacks import TensorBoard
tensorboard = TensorBoard(log_dir='GraphSingle', histogram_freq=0,  
          write_graph=True, write_images=True)
model = Alex().build()
tensorboard.set_model(model)

