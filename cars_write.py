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

FLAGS = tf.app.flags.FLAGS
  
tf.app.flags.DEFINE_string('directory', '/Users/gdutllc/Desktop/WorkSpace/',
                           'Directory to download data files and write the '
                           'converted result')

#%%
# 训练集样本的标签 8144
annotationFile = '../cars_anno/cars_train_annos.mat'
datamat = scio.loadmat(annotationFile)
annotations = datamat['annotations'][0]

# 标签整体类别 196
metaFile = '../cars_anno/cars_meta.mat'
metamat = scio.loadmat(metaFile)
meta = metamat['class_names'][0]

#%%

def convert_to(name):
    filename = os.path.join(FLAGS.directory, name + '.tfrecords')
    writer = tf.python_io.TFRecordWriter(filename)
        
    for num in range(len(annotations)):  
        label = get_annotation(num)
        img_name = label[5]
        img_path = FLAGS.directory + 'cars_img_train/' + img_name
        
        img = Image.open(img_path)
        
        if len(img.layer) < 3:
            img = img.convert("RGB")
            
        img = img.crop(label[:4])
        img = img.resize((224, 224))
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
# 每张图片自带的信息
def get_annotation(num):
	annotation = annotations[num]
	l = []
	for i in range(5):
		l.append(annotation[i][0][0])
	l.append(annotation[5][0])
	return tuple(l)


# 根据图片自带的信息查找meta，得到图片的标签
def splite_label(class_num):
    try:
    	result = meta[class_num-1][0].split(' ')
    	car_brand = result[0]
    	car_type = result[-2]
    	car_year = result[-1]
    	return car_brand, car_type, car_year
    
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

if __name__ == "__main__":
    
    convert_to('train_img')
    
    

        






        