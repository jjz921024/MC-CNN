#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar 31 16:03:15 2018

@author: gdutllc
"""
from keras.models import Model
from keras.layers import Dense,Flatten,Dropout,Input
from keras.layers.convolutional import Conv2D,MaxPooling2D

class Alex:
    def build(self):   
        input = Input(shape=(224,224,3), name='input_img')
        
        mid = Conv2D(96,(11,11),strides=(4,4),padding='valid',
               activation='relu',kernel_initializer='uniform')(input)
        mid = MaxPooling2D(pool_size=(3,3),strides=(2,2))(mid)
        mid = Conv2D(256,(5,5),strides=(1,1),padding='same',
                     activation='relu',kernel_initializer='uniform')(mid)
        mid = MaxPooling2D(pool_size=(3,3),strides=(2,2))(mid)
        mid = Conv2D(384,(3,3),strides=(1,1),padding='same',
                     activation='relu',kernel_initializer='uniform')(mid)
        mid = Conv2D(384,(3,3),strides=(1,1),padding='same',
                              activation='relu',kernel_initializer='uniform')(mid)
        mid = Conv2D(256,(3,3),strides=(1,1),padding='same',
                              activation='relu',kernel_initializer='uniform')(mid)
        mid = MaxPooling2D(pool_size=(3,3),strides=(2,2))(mid)
        
        mid = Flatten()(mid)
        
        #brand
        mid_brand = Dense(4096,activation='relu')(mid)
        mid_brand = Dropout(0.5)(mid_brand)
        mid_brand = Dense(4096,activation='relu')(mid_brand)
        mid_brand = Dropout(0.5)(mid_brand)
        output_brand = Dense(49,activation='softmax', name='output_brand')(mid_brand)
        
        #class
        mid_class = Dense(4096,activation='relu')(mid)
        mid_class = Dropout(0.5)(mid_class)
        mid_class = Dense(4096,activation='relu')(mid_class)
        mid_class = Dropout(0.5)(mid_class)
        output_class = Dense(22,activation='softmax', name='output_class')(mid_class)
        
        #year
        mid_year = Dense(4096,activation='relu')(mid)
        mid_year = Dropout(0.5)(mid_year)
        mid_year = Dense(4096,activation='relu')(mid_year)
        mid_year = Dropout(0.5)(mid_year)
        output_year = Dense(16,activation='softmax', name='output_year')(mid_year)
        
        self.model = Model(inputs=[input], outputs=[output_brand, output_class ,output_year])
        self.model.compile(loss='categorical_crossentropy',optimizer='sgd',metrics=['acc'])
        return self.model
        
#%%
"""
from keras.callbacks import TensorBoard
tensorboard = TensorBoard(log_dir='Graph_Mult', histogram_freq=0,  
          write_graph=True, write_images=True)

model = Alex().build()
tensorboard.set_model(model)
"""

