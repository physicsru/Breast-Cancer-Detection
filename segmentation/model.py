# -*- coding: utf-8 -*-
"""
Created on Thu Mar 15 16:10:24 2018

@author: System_Error
"""

import tensorflow as tf
import tensorflow.contrib.slim as slim



def UNet_same_padding(input_layer, index=1):
    #w, h = input_layer.get_shape().as_list()[1:3]
    conv1_1 = slim.conv2d(input_layer, 32*index, [3, 3], activation_fn=tf.nn.relu)
    conv1_2 = slim.conv2d(conv1_1, 32*index, [3, 3], activation_fn=tf.nn.relu)
    pool1 = tf.nn.max_pool(conv1_2, ksize = (1, 3, 3, 1), 
                           strides=(1, 2, 2, 1), padding='SAME')
    
    #1/2 size of original
    conv_down1_1 = slim.conv2d(pool1, 64*index, [3, 3], activation_fn=tf.nn.relu)
    conv_down1_2 = slim.conv2d(conv_down1_1, 64*index, [3, 3], activation_fn=tf.nn.relu)
    pool2 = tf.nn.max_pool(conv_down1_2, ksize = (1, 3, 3, 1), 
                           strides=(1, 2, 2, 1), padding='SAME')
    
    #1/4 size of original
    conv_down2_1 = slim.conv2d(pool2, 128*index, [3, 3], activation_fn=tf.nn.relu)
    conv_down2_2 = slim.conv2d(conv_down2_1, 128*index, [3, 3], activation_fn=tf.nn.relu)
    pool3 = tf.nn.max_pool(conv_down2_2, ksize = (1, 3, 3, 1), 
                           strides=(1, 2, 2, 1), padding='SAME')
    
    #1/8 size of original
    conv_down3_1 = slim.conv2d(pool3, 256*index, [3, 3], activation_fn=tf.nn.relu)
    conv_down3_2 = slim.conv2d(conv_down3_1, 256*index, [3, 3], activation_fn=tf.nn.relu)
    pool4 = tf.nn.max_pool(conv_down3_2, ksize = (1, 3, 3, 1), 
                           strides=(1, 2, 2, 1), padding='SAME')
    #1/16 size of original
    conv_down4_1 = slim.conv2d(pool4, 512*index, [3, 3], activation_fn=tf.nn.relu)
    conv_down4_2 = slim.conv2d(conv_down4_1, 512*index, [3, 3], activation_fn=tf.nn.relu)
    upscale1 = slim.conv2d_transpose(conv_down4_2, 256*index, [3, 3], stride = [2, 2])
    
    #1/8 size of original
    concate_up1 = tf.concat([conv_down3_2, upscale1], axis=3)
    conv_up1_1 = slim.conv2d(concate_up1, 256*index, [3, 3], activation_fn=tf.nn.relu)
    conv_up1_2 = slim.conv2d(conv_up1_1, 256*index, [3, 3], activation_fn=tf.nn.relu)
    upscale2 = slim.conv2d_transpose(conv_up1_2, 128*index, [3, 3], stride = [2, 2])
    
    #1/4 size of original
    concate_up2 = tf.concat([conv_down2_2, upscale2], axis=3)
    conv_up2_1 = slim.conv2d(concate_up2, 128*index, [3, 3], activation_fn=tf.nn.relu)
    conv_up2_2 = slim.conv2d(conv_up2_1, 128*index, [3, 3], activation_fn=tf.nn.relu)
    upscale3 = slim.conv2d_transpose(conv_up2_2, 64*index, [3, 3], stride = [2, 2])
    
    
    #1/2 size of original
    concate_up3 = tf.concat([conv_down1_2, upscale3], axis=3)
    conv_up3_1 = slim.conv2d(concate_up3, 64*index, [3, 3], activation_fn=tf.nn.relu)
    conv_up3_2 = slim.conv2d(conv_up3_1, 64*index, [3, 3], activation_fn=tf.nn.relu)
    upscale4 = slim.conv2d_transpose(conv_up3_2, 32*index, [3, 3], stride = [2, 2])
    
    #1/2 size of original
    concate_up4 = tf.concat([conv1_2, upscale4], axis=3)
    conv_up4_1 = slim.conv2d(concate_up4, 32*index, [3, 3], activation_fn=tf.nn.relu)
    conv_up4_2 = slim.conv2d(conv_up4_1, 32*index, [3, 3], activation_fn=tf.nn.relu)
    output = slim.conv2d(conv_up4_2, 3, [3, 3], activation_fn=None)
    return output
    

def UNet_simple(input_layer):
    #w, h = input_layer.get_shape().as_list()[1:3]
    conv1_1 = slim.conv2d(input_layer, 64, [3, 3], activation_fn=tf.nn.relu)
    conv1_2 = slim.conv2d(conv1_1, 64, [3, 3], activation_fn=tf.nn.relu)
    pool1 = tf.nn.max_pool(conv1_2, ksize = (1, 3, 3, 1), 
                           strides=(1, 2, 2, 1), padding='SAME')
    
    #1/2 size of original
    conv_down1_1 = slim.conv2d(pool1, 128, [3, 3], activation_fn=tf.nn.relu)
    conv_down1_2 = slim.conv2d(conv_down1_1, 128, [3, 3], activation_fn=tf.nn.relu)
    pool2 = tf.nn.max_pool(conv_down1_2, ksize = (1, 3, 3, 1), 
                           strides=(1, 2, 2, 1), padding='SAME')
    
    #1/4 size of original
    conv_down2_1 = slim.conv2d(pool2, 256, [3, 3], activation_fn=tf.nn.relu)
    conv_down2_2 = slim.conv2d(conv_down2_1, 256, [3, 3], activation_fn=tf.nn.relu)
    pool3 = tf.nn.max_pool(conv_down2_2, ksize = (1, 3, 3, 1), 
                           strides=(1, 2, 2, 1), padding='SAME')
    
    #1/8 size of original
    conv_down3_1 = slim.conv2d(pool3, 512, [3, 3], activation_fn=tf.nn.relu)
    conv_down3_2 = slim.conv2d(conv_down3_1, 512, [3, 3], activation_fn=tf.nn.relu)
    pool4 = tf.nn.max_pool(conv_down3_2, ksize = (1, 3, 3, 1), 
                           strides=(1, 2, 2, 1), padding='SAME')
    #1/16 size of original
    conv_down4_1 = slim.conv2d(pool4, 1024, [3, 3], activation_fn=tf.nn.relu)
    conv_down4_2 = slim.conv2d(conv_down4_1, 1024, [3, 3], activation_fn=tf.nn.relu)
    #upscale1 = upsample_conv(conv_down4_2, 512)
    upscale1 = slim.conv2d_transpose(conv_down4_2, 512, [3, 3], stride = [2, 2])
    
    #1/8 size of original
    concate_up1 = tf.concat([conv_down3_2, upscale1], axis=3)
    conv_up1_1 = slim.conv2d(concate_up1, 512, [3, 3], activation_fn=tf.nn.relu)
    conv_up1_2 = slim.conv2d(conv_up1_1, 512, [3, 3], activation_fn=tf.nn.relu)
    #upscale2 = upsample_conv(conv_up1_2, 256)
    upscale2 = slim.conv2d_transpose(conv_up1_2, 256, [3, 3], stride = [2, 2])
    
    #1/4 size of original
    concate_up2 = tf.concat([conv_down2_2, upscale2], axis=3)
    conv_up2_1 = slim.conv2d(concate_up2, 256, [3, 3], activation_fn=tf.nn.relu)
    conv_up2_2 = slim.conv2d(conv_up2_1, 256, [3, 3], activation_fn=tf.nn.relu)
    #upscale3 = upsample_conv(conv_up2_2, 128)
    upscale3 = slim.conv2d_transpose(conv_up2_2, 128, [3, 3], stride = [2, 2])
    
    
    #1/2 size of original
    concate_up3 = tf.concat([conv_down1_2, upscale3], axis=3)
    conv_up3_1 = slim.conv2d(concate_up3, 128, [3, 3], activation_fn=tf.nn.relu)
    conv_up3_2 = slim.conv2d(conv_up3_1, 128, [3, 3], activation_fn=tf.nn.relu)
    #upscale4 = upsample_conv(conv_up3_2, 64)
    upscale4 = slim.conv2d_transpose(conv_up3_2, 64, [3, 3], stride = [2, 2])
    
    #1/2 size of original
    concate_up4 = tf.concat([conv1_2, upscale4], axis=3)
    conv_up4_1 = slim.conv2d(concate_up4, 64, [3, 3], activation_fn=tf.nn.relu)
    conv_up4_2 = slim.conv2d(conv_up4_1, 64, [3, 3], activation_fn=tf.nn.relu)
    output = slim.conv2d(conv_up4_2, 3, [1, 1], activation_fn=None)
    #print(output.get_shape())
    return output
    

