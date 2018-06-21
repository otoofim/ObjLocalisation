import os
import io
import glob
import hashlib
import pandas as pd
import xml.etree.ElementTree as ET
import tensorflow as tf
import random
import cv2
import numpy as np
from PIL import Image




#This function assigns a specific digit to every class of objects.
def class_text_to_int(row_label):
        
    switcher = {
        
        "person": 1,
        "bird": 2,
        "cat": 3,
        "cow": 4,
        "dog": 5,
        "horse": 6,
        "sheep": 7,
        "aeroplane": 8,
        "bicycle": 9,
        "boat": 10,
        "bus": 11,
        "car": 12,
        "motorbike": 13,
        "train": 14,
        "bottle": 15,
        "diningtable": 16,
        "pottedplant": 17,
        "sofa": 18,
        "tvmonitor": 19,
        "chair": 20
        
    }
        
    if row_label in switcher.keys():
        return switcher.get(row_label)
    else:
        raise ValueError('The class is not defined: {0}'.format(row_label))



#This function converts an image to string so later on it can be used to be stored as tfdata
def load_image(addr):
    
    img = np.array(Image.open(addr))    
    return img.tostring()

#Returns tf variables 
def int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

#Returns tf variables 
def int64_list_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))

#Returns tf variables 
def bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

#Returns tf variables 
def bytes_list_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=value))

#Returns tf variables 
def float_list_feature(value):
    return tf.train.Feature(float_list=tf.train.FloatList(value=value))



def create_example(xml_file):

    #Loading xml file
    tree = ET.parse(xml_file)
    root = tree.getroot()
    
    #Getting image name and its dimentions
    image_name = root.find('filename').text
    file_name = image_name.encode('utf8')
    size=root.find('size')
    width = int(size[0].text)
    height = int(size[1].text)
    
    #Initilizing variables for the loaded image
    #Please note that following variables are vector. Since an image might have more than one object a vector is used.
    #So, forexample if an image includes 5 objects xmin will hold five numberse corresponding to each of the objects.
    xmin = []
    ymin = []
    xmax = []
    ymax = []
    classes = []
    classes_text = []
    truncated = []
    poses = []
    difficult_obj = []
    
    #Reading objects in an image one by one
    for member in root.findall('object'):
        
        #Adding name of the object
        classes_text.append(member.find("name").text)
        
        #Adding boundary of the object
        boundBox = member.find("bndbox")
        xmin.append(float(boundBox[0].text))
        ymin.append(float(boundBox[1].text))
        xmax.append(float(boundBox[2].text))
        ymax.append(float(boundBox[3].text))

        #Adding difficulty attributes. It means how far it is difficult to recognise the object
        difficult_obj.append(int(member.find("difficult").text))

        classes.append(class_text_to_int(member.find("name").text)) 
        
        # truncated feature is ommited becuase it nolonger availible in VOC2012 dataset.
        #To keep the dataset consistent, it is removed.
        #However, you can add it again just by uncomenting the line below and its corresponding part in building and restoring tfrecord.
        #truncated.append(int(member.find("truncated").text))
        
        #Adding position of the object that the image is taken.
        poses.append(member.find("pose").text)
    
    #Finding the corresponding image and turnining it to a string.
    full_path = os.path.join('../VOC2012/JPEGImages', '{}'.format(image_name))
    img = load_image(full_path)

    #Creating a new tfrecord from the above loaded features. 
    example = tf.train.Example(features=tf.train.Features(feature={
            'image_height': int64_feature(height),
            'image_width': int64_feature(width),
            'image_filename': bytes_feature(file_name),
            'image': bytes_list_feature(img),
            'xmin': float_list_feature(xmin),
            'xmax': float_list_feature(xmax),
            'ymin': float_list_feature(ymin),
            'ymax': float_list_feature(ymax),
            'classes': bytes_list_feature(classes_text),
            'label': int64_list_feature(classes),
            'difficult': int64_list_feature(difficult_obj),
            #'truncated': int64_list_feature(truncated),
            'view': bytes_list_feature(poses),
            }))
    
    return example


    #writer = tf.python_io.TFRecordWriter("../Tfdata/mytest.record")
    #writer.write(example.SerializeToString())
    #writer.close()




def run(xml_path, destination_path):
    #Provide path to write tfrecords there
    writer_train = tf.python_io.TFRecordWriter(destination_path+'train.record')     
    writer_test = tf.python_io.TFRecordWriter(destination_path+'test.record')
    
    #provide the path to annotation xml files directory
    filename_list=tf.train.match_filenames_once(xml_path)
    init = (tf.global_variables_initializer(), tf.local_variables_initializer())
    
    sess=tf.Session()
    sess.run(init)
    
    #shuffle files list
    list=sess.run(filename_list)
    random.shuffle(list)   
    
    i=1 
    tst=0   #to count number of images for evaluation 
    trn=0   #to count number of images for training
    
    for xml_file in list:
        print(xml_file)
        #Create a tfrecord
        example = create_example(xml_file)
        
        #each 10th file (xml and image) write it for evaluation
        if (i%10)==0: 
            writer_test.write(example.SerializeToString())
            tst=tst+1
        #the rest for training
        else:          
            writer_train.write(example.SerializeToString())
            trn=trn+1
        i=i+1
        
    writer_test.close()
    writer_train.close()
    
    print('Successfully converted dataset to TFRecord.')
    print('training dataset: # ')
    print(trn)
    print('test dataset: # ')
    print(tst)


# http://warmspringwinds.github.io/tensorflow/tf-slim/2016/12/21/tfrecords-guide/
# http://machinelearninguru.com/deep_learning/tensorflow/basics/tfrecord/tfrecord.html

# training dataset: # 
# 
# 15413
# 
# test dataset: # 
# 
# 1712
