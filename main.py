#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr  6 10:46:09 2018

@author: jun
"""
import tensorflow as tf
#from alexnet_single import Alex as Alex_single
from alexnet_mult import Alex as Alex_mult
from keras import backend as K
#from keras.objectives import categorical_crossentropy
#from keras.metrics import categorical_accuracy as accuracy
from data_load import batch_input
from keras.callbacks import EarlyStopping

#%%
imgBatch, labelBatch = batch_input("train_img.tfrecords", batchSize=128, num_epochs=50)

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
model = Alex_mult().build() #input_img

#output_label = tf.placeholder(tf.float32, shape=(None, 49))
#loss = tf.reduce_mean(categorical_crossentropy(output_label, model.output))


#es = EarlyStopping(monitor='loss', patience=5, verbose=1)



#%%
try:
    step = 0
    while not coord.should_stop():
        step = step + 1
        example, label_brand, label_class, label_year = sess.run([
                imgBatch, labelBrand_onehot, labelClass_onehot, labelYear_onehot])   
    
        #single
        #loss_and_metrics = model.train_on_batch(example, label_brand)
         
        
        #mult
        loss_and_metrics = model.train_on_batch({'input_img' : example},
                                                {'output_brand' : label_brand, 
                                                 'output_class' : label_class, 
                                                 'output_year' : label_year})
                                           
                
        if step % 200 == 0:
            #loss_and_metrics = model.test_on_batch(example, label_brand)
            print("loss: ",loss_and_metrics[0], " acc: ", loss_and_metrics[1])
      
        
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
                    output_label : label}))
        
        #model.metrics_names   fit可以知道loss_and_metrice内容
        """

   
        

except tf.errors.OutOfRangeError:
    print('Done training -- epoch limit reached')
finally:
    model.save_weights('mult_alexnet_weights.h5')
    coord.request_stop()


coord.join(threads)
sess.close()