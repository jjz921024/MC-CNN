#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr  1 19:48:53 2018

@author: gdutllc
"""
import tensorflow as tf

#%%
def read_data(fileNameQue):
    reader = tf.TFRecordReader()
    key, value = reader.read(fileNameQue)
    features = tf.parse_single_example(value, features={
                                    'label_brand': tf.FixedLenFeature([], tf.int64),
                                    'label_class': tf.FixedLenFeature([], tf.int64), 
                                    'label_year': tf.FixedLenFeature([], tf.int64), 
                                    'img': tf.FixedLenFeature([], tf.string)
                                    })
    
    #img = tf.image.decode_image(features['img'], channels=3)
    img = tf.decode_raw(features['img'], tf.uint8)
    
    img = tf.reshape(img, [227, 227, 3])
    img.set_shape([227, 227, 3])
    img = tf.cast(img, tf.float32) * (1. / 255) - 0.5
    
    label = []
    label.append(tf.cast(features['label_brand'], tf.int32))
    label.append(tf.cast(features['label_class'], tf.int32))
    label.append(tf.cast(features['label_year'], tf.int32))
    return img, label


def batch_input(filename, batchSize, num_epochs):
    fileNameQue = tf.train.string_input_producer([filename], shuffle=True, num_epochs=num_epochs)
    img, label = read_data(fileNameQue) 
    min_after_dequeue = 1000
    capacity = min_after_dequeue + 3 * batchSize
    
    # img = tf.image.resize_images(img, [224, 224])
    # 预取图像和label并随机打乱，组成batch，此时tensor rank发生了变化，多了一个batch大小的维度
    #exampleBatch, labelBatch = tf.train.shuffle_batch([img, label],batch_size=100, capacity=1300,min_after_dequeue=1000)
    exampleBatch, labelBatch = tf.train.shuffle_batch([img, label],
                                                     batch_size=batchSize, capacity=capacity,
                                                     num_threads=2,
                                                     min_after_dequeue=min_after_dequeue)
    return exampleBatch, labelBatch



#%%
"""
for serialized_example in tf.python_io.tf_record_iterator("train.tfrecords"):
    example = tf.train.Example()
    example.ParseFromString(serialized_example)
    
    image = example.features.feature['img'].bytes_list.value
    image = tf.reshape(image, [224, 224, 3])
    image = tf.cast(image, tf.float32) * (1. / 255) - 0.5
   
    plt.figure("car")
    plt.imshow(image)
    plt.show()
    #print(image, label)
    break
"""


