#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr  6 17:14:29 2018

@author: jun
"""
import numpy as np
import tensorflow as tf
#from keras.models import model_from_json  
#import matplotlib.pyplot as plt 
#import matplotlib.image as mpimg
from PIL import Image
from alexnet_single import Alex as Alex_single
from data_load import batch_input
from keras import backend as K
from cars_write import result2label
from file_write import write_result

#读取model  
#model=model_from_json(open('single_alexnet_architecture.json').read())  
model = Alex_single().build()
model.load_weights('single_alexnet_weights.h5')  

#%%
'''
path_test = '/home/jun/dataset/cars_test'

img = Image.open(path_test + '/00001.jpg')
if len(img.layer) < 3:
    img = img.convert("RGB")

img = img.resize((224, 224))
image_array = np.array(img)[:, :, 0:3]
image_array = image_array.reshape(1,224,224,3)

#img = tf.cast(img, tf.float32) * (1. / 255) - 0.5
pre=model.predict_classes(image_array)
print(pre)
'''
#%%
imgBatch, labelBatch = batch_input("train_img.tfrecords", batchSize=100, num_epochs=1)

labelBatch = tf.transpose(labelBatch, perm=[1, 0])

local_init = tf.local_variables_initializer()
init = tf.global_variables_initializer()

sess = tf.Session()
sess.run(local_init)
sess.run(init)

coord = tf.train.Coordinator()
threads = tf.train.start_queue_runners(sess=sess, coord=coord)

K.set_session(sess)

#%%
try:
    while not coord.should_stop():
        example, label = sess.run([imgBatch, labelBatch])   
    
        pred = model.predict_on_batch(example)
        pred = np.argmax(pred,axis=1)
        
        trust_brand = result2label(label[0])
        pred_brand = result2label(pred)
        #pred_brand = onehot2label(tf.concat([tf.one_hot(pred.transpose[0], 49, 1, 0)], axis=1), sess=sess)
        #trust_brand = onehot2label(tf.concat([label_brand], axis=1), sess=sess)
        write_result(trust_brand, pred_brand)
        

except tf.errors.OutOfRangeError:
    print('Done training -- epoch limit reached')
finally:
    coord.request_stop()


coord.join(threads)
sess.close()

