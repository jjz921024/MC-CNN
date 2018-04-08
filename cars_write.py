#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar 31 21:16:12 2018

@author: gdutllc
"""
import tensorflow as tf
import os
import scipy.io as scio
from PIL import Image
import numpy as np

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string('directory', '/home/jun/PythonWorkspace/MC-CNN',
                           'Directory to download data files and write the '
                           'converted result')

#%%
# 训练集样本的标签 8144
annotationFile = '/home/jun/dataset/cars_anno/cars_train_annos.mat'
datamat = scio.loadmat(annotationFile)
annotations = datamat['annotations'][0]

annotationFile_test = '/home/jun/dataset/cars_anno/cars_test_annos.mat'  
datamat_test = scio.loadmat(annotationFile_test)
annotations_test = datamat_test['annotations'][0]

# 标签整体类别 196
metaFile = '/home/jun/dataset/cars_anno/cars_meta.mat'
metamat = scio.loadmat(metaFile)
meta = metamat['class_names'][0]

#%%

def convert_to(name, isTrain):
    filename = os.path.join(FLAGS.directory, name + '.tfrecords')
    writer = tf.python_io.TFRecordWriter(filename)
    
    
    for num in range(len(annotations) if isTrain else len(annotations_test)):  
        label = get_annotation(num, isTrain)
        img_name = label[5]
        if isTrain:
            img_path = '/home/jun/dataset/cars_train/' + img_name   
        else:
            img_path = '/home/jun/dataset/cars_test/' + img_name
        
        img = Image.open(img_path)
        
        if len(img.layer) < 3:
            img = img.convert("RGB")
            
        img = img.crop(label[:4])
        img = img.resize((227, 227))
        img_raw = img.tobytes()  
        
        brand, classes, year = splite_label(label[4])
        
        example = tf.train.Example(features=tf.train.Features(
                feature={
                        'label_brand': tf.train.Feature(int64_list=tf.train.Int64List(value=[brand_list.index(brand)])),
                        'label_class': tf.train.Feature(int64_list=tf.train.Int64List(value=[class_list.index(classes)])),
                        'label_year': tf.train.Feature(int64_list=tf.train.Int64List(value=[year_list.index(year)])),
                        'img': tf.train.Feature(bytes_list=tf.train.BytesList(value=[img_raw]))
                        #'img': tf.train.Feature(bytes_list=tf.image.encode_jpeg(np.array(img), format='rgb'))
                    }))
        writer.write(example.SerializeToString())  
    writer.close()
#%%   
# 获得标注  tuple(x1,y1,x2,y2, class, fileName)
# 每张图片自带的信息, test-set is not provide class
def get_annotation(num, isTrain):
    l = []
    if isTrain:
        annotation = annotations[num]
        for i in range(5):
            l.append(annotation[i][0][0])
        l.append(annotation[5][0])
    else:
        annotation = annotations_test[num]
        for i in range(4):
            l.append(annotation[i][0][0])
        l.append(-1)
        l.append(annotation[4][0])
    return tuple(l)


# 根据图片自带的信息查找meta，得到图片的标签
def splite_label(class_num):
    try:
    	result = meta[class_num-1][0].split(' ')
    	car_brand = result[0]
    	car_class = result[-2]
    	car_year = result[-1]
    	return car_brand, car_class, car_year
    
    except IndexError:
        print("error!!!", class_num)
        
        

#%%
brand_set = set()
class_set = set()
year_set = set()
for i in range(len(meta)):
	result = meta[i][0].split(' ')
	brand_set.add(result[0])
	class_set.add(result[-2])
	year_set.add(result[-1])
brand_list = list(brand_set)
class_list = list(class_set)
year_list = list(year_set)


#%%
def onehot2label(output, sess):
    result_brand = []
    result_classes = []
    result_year = []
    for i in range(output.shape[0]):
        decode1, decode2, decode3 = np.split(output, [49,71], axis=1)
        #decode1 = tf.argmax(tf.slice(output,[0,0],[100,49]), axis=1)
        #decode2 = tf.argmax(tf.slice(output,[0,49],[100,22]), axis=1)
        #decode3 = tf.argmax(tf.slice(output,[0,71],[100,16]), axis=1)
    
        #result1, result2, result3 = sess.run([decode1, decode2, decode3])
        #result1 = sess.run([decode1])
        
        #result_brand.append(brand_list[result1[i]])
        #result_classes.append(class_list[result2[i]])
        #result_year.append(year_list[result3[i]])
    return result_brand, result_classes, result_year

def result2label(output):
    result_brand = []
    for i in range(output.shape[0]):
        result_brand.append(brand_list[output[i]])
    return result_brand


#%%

#if __name__ == "__main__":
#convert_to('test_img', False)
    
    

        






        