# -*- coding: utf-8 -*-
"""
Created on Thu Mar 22 10:46:07 2018

@author: System_Error
"""

import tensorflow as tf
import numpy as np
import cv2
import os
#from keras.utils import to_categorical
#from sklearn.preprocessing import OneHotEncoder



def load_data(image_dir):
    images = []
    for name in os.listdir(image_dir):
        file_name = os.path.join(image_dir, name)
        image = cv2.imread(file_name, 0)
        images.append(image)
        
    return np.asarray(images)


def load_label(image_dir):
    images = []
    for name in os.listdir(image_dir):
        file_name = os.path.join(image_dir, name)
        image = cv2.imread(file_name)
        images.append(image)
        
    return np.asarray(images)


def load_test_data(image_dir):
    images = []
    for name in os.listdir(image_dir):
        file_name = os.path.join(image_dir, name)
        image = cv2.imread(file_name, 0)
        image = np.expand_dims(image, -1)
        image = image[16: 496, 16:496, :]
        images.append(image)
        
    return np.asarray(images)


def next_batch(image, label, img_size, batch_size):
    img_list, label_list = [], []
    for i in range(batch_size):
        j = np.random.randint(0, len(image))
        img, lbl = augment(image[j], label[j], img_size)
        img_list.append(img)
        label_list.append(lbl)
    img_list = np.asarray(img_list)
    label_list = np.asarray(label_list)
    '''
    print(np.shape(img_list))
    print(np.shape(label_list))
    '''
    return img_list, label_list


def augment(image, label, img_size):

    h, w = np.shape(image)[0], np.shape(image)[1]
    h_size = int((0.8+0.2*np.random.ranf())*h)
    w_size = int((0.8+0.2*np.random.ranf())*w)
    offset_h = int((512-h)*np.random.ranf())
    offset_w = int((512-w)*np.random.ranf())
    
    image = image[offset_h: offset_h + h_size, offset_w: offset_w + w_size]
    label = label[offset_h: offset_h + h_size, offset_w: offset_w + w_size]
    image = cv2.resize(image, (img_size, img_size))
    label = cv2.resize(label, (img_size, img_size))
    flip_prop = np.random.randint(0, 100)
    if flip_prop > 50:
        image = cv2.flip(image, 1)
        label = cv2.flip(label, 1)
    image = np.reshape(image/255, (img_size, img_size, 1))
    label = np.reshape(label/255, (img_size, img_size, 3))
    return image, label


def label2one_hot(input_):
    onehot_encoder = OneHotEncoder(sparse=False)
    one_hot_vec = onehot_encoder.fit_transform(input_.reshape(-1, 1))
    return one_hot_vec


def dice_loss(y_true, y_pred):
    #print(y_true.get_shape())
    #print(y_pred.get_shape())
    h, w, c = y_true.get_shape()[1:]
    smooth = 1.0
    y_true_f = tf.reshape(y_true, (-1, h*w*c))
    y_pred_f = tf.reshape(y_pred, (-1, h*w*c))
    intersection = tf.reduce_mean(y_true_f * y_pred_f)
    member = (2. * intersection + smooth) 
    denominator = tf.reduce_sum(y_true_f) + tf.reduce_sum(y_pred_f) + smooth
    return -(member/denominator)
        

def dice_coe(target, output, loss_type='jaccard', axis=(1, 2, 3), smooth=1):
    inse = tf.reduce_sum(output * target, axis=axis)
    if loss_type == 'jaccard':
        l = tf.reduce_sum(output * output, axis=axis)
        r = tf.reduce_sum(target * target, axis=axis)
    elif loss_type == 'sorensen':
        l = tf.reduce_sum(output, axis=axis)
        r = tf.reduce_sum(target, axis=axis)
    else:
        raise Exception("Unknow loss_type")
    dice = (2. * inse + smooth) / (l + r + smooth)
    dice = tf.reduce_mean(dice)
    return dice

def ce_loss(target, pred):
    num_cls = target.get_shape().as_list()[-1]
    target = tf.reshape(target, [-1, num_cls])
    pred = tf.reshape(pred, [-1, num_cls])
    softmax_pred = tf.nn.softmax(pred)
    ce = tf.losses.softmax_cross_entropy(target, softmax_pred)
    loss = tf.reduce_mean(ce)
    return loss
