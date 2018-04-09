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
num_classes_brand = 49
num_classes_classes = 22
num_classes_year = 16

#%%
imgBatch, labelBatch = batch_input("train_img.tfrecords", batchSize=batch_size, num_epochs=num_epochs)
imgBatch_test, labelBatch_test = batch_input("test_img.tfrecords", batchSize=batch_size, num_epochs=num_epochs)

sess = tf.Session()
sess.run(tf.local_variables_initializer()) #above coordinator

coord = tf.train.Coordinator()
threads = tf.train.start_queue_runners(sess=sess, coord=coord)

#转置 labelBatch的shape要为(3, batchSize)
labelBatch = tf.transpose(labelBatch, perm=[1, 0])
labelBatch_test = tf.transpose(labelBatch_test, perm=[1, 0])

#onehot的shape要为(batchSize, 49)
labelBrand_onehot = tf.one_hot(labelBatch[0], num_classes_brand, 1, 0)
labelClass_onehot = tf.one_hot(labelBatch[1], num_classes_classes, 1, 0)
labelYear_onehot = tf.one_hot(labelBatch[2], num_classes_year, 1, 0)

labelBrand_onehot_test = tf.one_hot(labelBatch_test[0], num_classes_brand, 1, 0)
labelClass_onehot_test = tf.one_hot(labelBatch_test[1], num_classes_classes, 1, 0)
labelYear_onehot_test = tf.one_hot(labelBatch_test[2], num_classes_year, 1, 0)


input_img = tf.placeholder(tf.float32, [batch_size, 227, 227, 3])
keep_prob = tf.placeholder(tf.float32)
y_brand = tf.placeholder(tf.float32, [batch_size, num_classes_brand])
y_classes = tf.placeholder(tf.float32, [batch_size, num_classes_classes])
y_year = tf.placeholder(tf.float32, [batch_size, num_classes_year])

model = AlexNet(input_img, num_classes_brand, num_classes_classes, num_classes_year, keep_prob)
score_brand = model.fc8_brand
score_classes = model.fc8_classes
score_year = model.fc8_year


with tf.name_scope("cross_ent_brand"):
    loss_brand = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=score_brand,labels=y_brand))
#y_pred=tf.nn.softmax(tf.matmul(output, output_w) + output_b)
#loss = tf.reduce_mean(-tf.reduce_sum(y_trust * tf.log(y_pred), reduction_indices=[1]))
    
with tf.name_scope("accuracy_brand"):
    correct_pred_brand = tf.equal(tf.argmax(score_brand, 1), tf.argmax(y_brand, 1))
    accuracy_brand = tf.reduce_mean(tf.cast(correct_pred_brand, tf.float32))
  

with tf.name_scope("cross_ent_classes"):
    loss_classes = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=score_classes,labels=y_classes))
    
with tf.name_scope("accuracy_classes"):
    correct_pred_classes = tf.equal(tf.argmax(score_classes, 1), tf.argmax(y_classes, 1))
    accuracy_classes = tf.reduce_mean(tf.cast(correct_pred_classes, tf.float32))
    
    
with tf.name_scope("cross_ent_year"):
    loss_year = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=score_year,labels=y_year))
    
with tf.name_scope("accuracy_brand"):
    correct_pred_year = tf.equal(tf.argmax(score_year, 1), tf.argmax(y_year, 1))
    accuracy_year = tf.reduce_mean(tf.cast(correct_pred_year, tf.float32))
    

train_op_brand = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss_brand)
train_op_classes = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss_classes)
train_op_year = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss_year)


tf.summary.scalar('loss_brand', loss_brand)
tf.summary.scalar('acc_brand', accuracy_brand)
tf.summary.scalar('loss_classes', loss_classes)
tf.summary.scalar('acc_classes', accuracy_classes)
tf.summary.scalar('loss_year', loss_year)
tf.summary.scalar('acc_year', accuracy_year)
merged_op = tf.summary.merge_all()

#saver = tf.train.Saver()

summary_writer_train = tf.summary.FileWriter('Graph_Mult/train')
summary_writer_test = tf.summary.FileWriter('Graph_Mult/test')

sess.run(tf.global_variables_initializer())

#%%
try:          
    step = 0
    while not coord.should_stop():
        step = step + 1
        example, label_brand, label_class, label_year = sess.run([
                imgBatch, labelBrand_onehot, labelClass_onehot, labelYear_onehot])   
        
        sess.run([train_op_brand, train_op_classes, train_op_year],
                 feed_dict={input_img:example, y_brand:label_brand, y_classes:label_class, y_year:label_year, keep_prob: dropout_rate})
        
                                           
                
        if step % 100 == 0:
            #test
            example_test, label_brand_test, label_class_test, label_year_test = sess.run([
                imgBatch_test, labelBrand_onehot_test, labelClass_onehot_test, labelYear_onehot_test])
    
            acc_b_test, ls_b_test, acc_c_test, ls_c_test, acc_y_test, ls_y_test, merged_train = sess.run([accuracy_brand, loss_brand, 
                                                                      accuracy_classes, loss_classes,
                                                                      accuracy_year, loss_year,
                                                                      merged_op], feed_dict={input_img:example_test, y_brand:label_brand_test, y_classes:label_class_test, y_year:label_year_test, keep_prob: 1})       
            
            summary_writer_test.add_summary(merged_train, step)
            summary_writer_test.flush()
            
            #train
            acc_b, ls_b, acc_c, ls_c, acc_y, ls_y, merged_test = sess.run([accuracy_brand, loss_brand, 
                                                                      accuracy_classes, loss_classes,
                                                                      accuracy_year, loss_year,
                                                                      merged_op], feed_dict={input_img:example, y_brand:label_brand, y_classes:label_class, y_year:label_year, keep_prob: 1})       
            
            print("step %d: acc_b: %g, loss_b: %g \n\t acc_c: %g, loss_c: %g \n\t acc_y: %g, loss_y: %g" 
                  %(step, acc_b, ls_b, acc_c, ls_c, acc_y, ls_y))
            
            summary_writer_train.add_summary(merged_test, step)
            summary_writer_train.flush()
            
   
        

except tf.errors.OutOfRangeError:
    print('Done training -- epoch limit reached')
finally:
    #save_path = saver.save(sess, 'tf_alexnet')
    coord.request_stop()


coord.join(threads)
sess.close()