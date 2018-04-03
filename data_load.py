#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr  1 19:48:53 2018

@author: gdutllc
"""
import tensorflow as tf
from alexnet_mult import Alex
from keras import backend as K
from keras.objectives import categorical_crossentropy
from keras.metrics import categorical_accuracy as accuracy

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
    
    img = tf.reshape(img, [224, 224, 3])
    img.set_shape([224, 224, 3])
    img = tf.cast(img, tf.float32) * (1. / 255) - 0.5
    
    label = []
    label.append(tf.cast(features['label_brand'], tf.int32))
    label.append(tf.cast(features['label_class'], tf.int32))
    label.append(tf.cast(features['label_year'], tf.int32))
    return img, label


def batch_input(filename, batchSize, num_epochs):
    #fileNameQue = tf.train.string_input_producer(['../train_img.tfrecords'])
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

#%%
imgBatch, labelBatch = batch_input("../train_img.tfrecords", batchSize=100, num_epochs=1)
#转置 labelBatch的shape要为(3, batchSize)
labelBatch = tf.transpose(labelBatch, perm=[1, 0])
#onehot的shape要为(batchSize, 49)
labelBrand_onehot = tf.one_hot(labelBatch[0], 49, 1, 0)
labelClass_onehot = tf.one_hot(labelBatch[1], 22, 1, 0)
labelYear_onehot = tf.one_hot(labelBatch[2], 16, 1, 0)

local_init = tf.local_variables_initializer()
init = tf.global_variables_initializer()

sess = tf.Session()
sess.run(local_init)
sess.run(init)

coord = tf.train.Coordinator()
threads = tf.train.start_queue_runners(sess=sess, coord=coord)

K.set_session(sess)
#input_img = tf.placeholder(tf.float32, shape=(None, 224, 224, 3))
model = Alex().build() #input_img

#output_label = tf.placeholder(tf.float32, shape=(None, 49))
#loss = tf.reduce_mean(categorical_crossentropy(output_label, model.output))




#%%
try:
    step = 0
    while not coord.should_stop():
        #step = step + 1
        example, label_brand, label_class, label_year = sess.run([
                imgBatch, labelBrand_onehot, labelClass_onehot, labelYear_onehot])
    
        loss_and_metrics = model.train_on_batch({'input_img' : example},
                                                {'output_brand' : label_brand,
                                                 'output_class' : label_class,
                                                 'output_year' : label_year})
        print("loss: ",loss_and_metrics[0], " acc: ", loss_and_metrics[1])
        
        """
        if step % 10 == 0:
            loss_and_metrics = model.evaluate(example, label, batch_size=10)
            print("loss: ",loss_and_metrics[0], " acc: ", loss_and_metrics[1])
        """
        
        
        
        """
        #tensorflow
        train_op = tf.train.GradientDescentOptimizer(0.5).minimize(loss)
        train_op.run(sess=sess, feed_dict={
                input_img : example,
                output_label : label})
    
        if step % 10 == 0:
            acc_value = accuracy(label, model.output)
            print(acc_value.eval(feed_dict={
                    input_img : example,
                    output_label : label}))"""
        
        #model.fit(example, label, batch_size=10, epochs=1)
    
        

except tf.errors.OutOfRangeError:
    print('Done training -- epoch limit reached')
finally:
    coord.request_stop()


coord.join(threads)
sess.close()





