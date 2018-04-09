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
imgBatch, labelBatch = batch_input("train_img.tfrecords", batchSize=128, num_epochs=100)

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
K.set_learning_phase(1)

model = Alex_mult().build() 
#loss = tf.reduce_mean(categorical_crossentropy(output_label, model.output))


#es = EarlyStopping(monitor='loss', patience=5, verbose=1)

summay_writer = tf.summary.FileWriter('Graph_Mult', sess.graph)
merged_op = tf.summary.merge_all()

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
                                           
                
        if step % 100 == 0:
            #loss_and_metrics = model.test_on_batch(example, label_brand)
            print("loss_total: ",loss_and_metrics[0], " brand_acc: ", loss_and_metrics[5], 
                  " class_acc: ", loss_and_metrics[7], " year_acc: ", loss_and_metrics[9])
            tf.summary.scalar('brand_loss', loss_and_metrics[1])
            tf.summary.scalar('class_loss', loss_and_metrics[2])
            tf.summary.scalar('year_loss', loss_and_metrics[3])
            tf.summary.scalar('brand_acc', loss_and_metrics[5])
            tf.summary.scalar('class_acc', loss_and_metrics[7])
            tf.summary.scalar('year_acc', loss_and_metrics[9])
            merged = sess.run(merged_op)
            summay_writer.add_summary(merged, step)
            
      
        
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