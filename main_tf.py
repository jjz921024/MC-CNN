#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr  6 10:46:09 2018

@author: jun
"""
import tensorflow as tf
from data_load import batch_input
from alexnet import AlexNet

# Learning params
learning_rate = 0.01
num_epochs = 150
batch_size = 128

# Network params
dropout_rate = 0.5
num_classes = 49

#%%
imgBatch, labelBatch = batch_input("train_img.tfrecords", batchSize=batch_size, num_epochs=num_epochs)

sess = tf.Session()
sess.run(tf.local_variables_initializer()) #above coordinator

coord = tf.train.Coordinator()
threads = tf.train.start_queue_runners(sess=sess, coord=coord)

#转置 labelBatch的shape要为(3, batchSize)
labelBatch = tf.transpose(labelBatch, perm=[1, 0])

#onehot的shape要为(batchSize, 49)
labelBrand_onehot = tf.one_hot(labelBatch[0], 49, 1, 0)
labelClass_onehot = tf.one_hot(labelBatch[1], 22, 1, 0)
labelYear_onehot = tf.one_hot(labelBatch[2], 16, 1, 0)


input_img = tf.placeholder(tf.float32, [batch_size, 224, 224, 3])
keep_prob = tf.placeholder(tf.float32)
y = tf.placeholder(tf.float32, [batch_size, num_classes])

model = AlexNet(input_img, num_classes, keep_prob)
score = model.fc8


with tf.name_scope("cross_ent"):
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=score,labels=y))
#y_pred=tf.nn.softmax(tf.matmul(output, output_w) + output_b)
#loss = tf.reduce_mean(-tf.reduce_sum(y_trust * tf.log(y_pred), reduction_indices=[1]))
    
with tf.name_scope("accuracy"):
    correct_pred = tf.equal(tf.argmax(score, 1), tf.argmax(y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

train_op = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)


tf.summary.scalar('loss', loss)
tf.summary.scalar('acc', accuracy)
merged_op = tf.summary.merge_all()

saver = tf.train.Saver()

summary_writer = tf.summary.FileWriter('Graph_TF')

sess.run(tf.global_variables_initializer())

#%%
try:          
    step = 0
    while not coord.should_stop():
        step = step + 1
        example, label_brand, label_class, label_year = sess.run([
                imgBatch, labelBrand_onehot, labelClass_onehot, labelYear_onehot])   
        
        sess.run([train_op],feed_dict={input_img:example, y:label_brand, keep_prob: dropout_rate})
        
                                           
                
        if step % 100 == 0:
            acc, ls, merged = sess.run([accuracy, loss, merged_op], 
                                         feed_dict={input_img:example, y:label_brand, keep_prob: 1})       
            print("step %d, acc %g, loss: %g" %(step, acc, ls))
            
            summary_writer.add_summary(merged, step)
            summary_writer.flush()
            
   
        

except tf.errors.OutOfRangeError:
    print('Done training -- epoch limit reached')
finally:
    save_path = saver.save(sess, 'tf_alexnet')
    coord.request_stop()


coord.join(threads)
sess.close()