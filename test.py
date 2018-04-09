#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr  9 10:55:03 2018

@author: jjz
"""

import tensorflow as tf
from data_load import batch_input


num_epochs = 150
batch_size = 128



#%%
imgBatch, labelBatch = batch_input("train_img.tfrecords", batchSize=batch_size, num_epochs=num_epochs)
imgBatch_test, labelBatch_test = batch_input("test_img.tfrecords", batchSize=batch_size, num_epochs=num_epochs)

sess = tf.Session()
sess.run(tf.local_variables_initializer()) #above coordinator

coord = tf.train.Coordinator()
threads = tf.train.start_queue_runners(sess=sess, coord=coord)

#%%
try:          
    while not coord.should_stop():
        example = sess.run([imgBatch, imgBatch_test])   
        
       
            
   
        

except tf.errors.OutOfRangeError:
    print('Done training -- epoch limit reached')
finally:
    coord.request_stop()