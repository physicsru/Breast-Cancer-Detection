# -*- coding: utf-8 -*-
"""
Created on Thu Mar 15 17:45:07 2018

@author: System_Error
"""

import tensorflow as tf
import numpy as np
import time
import utils
import os
import cv2
import argparse
from model import UNet_same_padding
from skimage import exposure


def arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--img_size", default = 480, type = int)
    parser.add_argument("--batch_size", default = 8, type = int)
    parser.add_argument("--iterations", default = 50000, type = int)
    parser.add_argument("--init_lr", default = 1e-4, type = float)
    parser.add_argument("--train_data_dir", default = 'dataset/data')
    parser.add_argument("--train_label_dir", default = 'dataset/label_rgb')
    parser.add_argument("--save_dir", default = 'saved_models')
    parser.add_argument("--out_dir", default = 'output')
    parser.add_argument("--mode", default = 'test')
    parser.add_argument("--gpu_fraction", default = 0.4, type = float)
    args = parser.parse_args()
    return args


class UNet():
    def __init__(self, args):
        self.img_size = args.img_size
        self.batch_size = args.batch_size
        self.iterations = args.iterations
        self.init_lr = args.init_lr
        self.data_dir = args.train_data_dir
        self.label_dir = args.train_label_dir
        self.save_dir = args.save_dir
        self.out_dir = args.out_dir
        self.gpu_fraction = args.gpu_fraction
    
    
    def build_network(self):
        self.input = tf.placeholder(tf.float32, 
                        [None, self.img_size, self.img_size, 1], 'data')
        self.label = tf.placeholder(tf.float32, 
                        [None, self.img_size, self.img_size, 3], 'label')
        self.learning_rate = tf.placeholder(tf.float32)
        self.output = UNet_same_padding(self.input)
        
        '''
        flatten_out = tf.reshape(self.output, shape=[-1, self.img_size * self.img_size, 2])
        softmax_out = tf.nn.softmax(flatten_out)
        self.loss = tf.losses.softmax_cross_entropy(self.label, softmax_out)
        '''
        
        self.loss = utils.ce_loss(self.label, self.output)
        self.optimizer = tf.train.AdamOptimizer(self.learning_rate).minimize(self.loss)
        tf.summary.scalar("Loss", self.loss)
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=self.gpu_fraction)
        self.sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
        self.saver = tf.train.Saver()
    
    
    def load_train_data(self):
        self.train_data = utils.load_data(self.data_dir)
        self.train_label = utils.load_label(self.label_dir)
        #print(np.shape(self.train_data))
        #print(np.shape(self.train_label))
    
    
    def load_test_data(self):
        self.test_data = utils.load_test_data('dataset/test_data')
        
    
    
    def train(self):
        if not os.path.exists(self.save_dir):
            os.mkdir(self.save_dir)

        init = tf.global_variables_initializer()
        summary_op = tf.summary.merge_all()
        #onehot_encoder = OneHotEncoder(sparse=False)
        train_writer = tf.summary.FileWriter(self.save_dir+"/train", self.sess.graph)
        self.sess.run(init)
        lr = self.init_lr
        start = time.time()
        
        for i in range(self.iterations):
            
            batch_data, batch_label = utils.next_batch(self.train_data, self.train_label, 
                                                        self.img_size, self.batch_size)

            summary, loss, _ = self.sess.run([summary_op, self.loss, self.optimizer], 
                                             feed_dict = {self.input: batch_data, 
                                                          self.label: batch_label, 
                                                          self.learning_rate: lr})
            train_writer.add_summary(summary, i)
            #time.sleep(0.1)
            
            if np.mod(i+1, 10) == 0:
                print('iter:', i+1, 'loss:', loss, 'time:', time.time()-start)
                start = time.time()
        
            if np.mod(i+1, 10000) == 0:
                self.saver.save(self.sess, self.save_dir+"/model", global_step=i)
            
        
        
    def test(self):
        if not os.path.exists(self.out_dir):
            os.mkdir(self.out_dir)
        self.saver.restore(self.sess, tf.train.latest_checkpoint(self.save_dir))
        for i in range(len(self.test_data)):
            #print(np.shape(self.test_data[i]))
            img = np.expand_dims(self.test_data[i], axis=0)
            out_img = self.sess.run([self.output], feed_dict = {self.input: img})
            
            out_img = np.squeeze(out_img)
            out_img = exposure.rescale_intensity(out_img, out_range=(0, 255))
            
            cv2.imwrite(self.out_dir+'/out_r'+str(i).zfill(4)+'.bmp', out_img[:, :, 0])
            cv2.imwrite(self.out_dir+'/out_g'+str(i).zfill(4)+'.bmp', out_img[:, :, 1])
            cv2.imwrite(self.out_dir+'/out_b'+str(i).zfill(4)+'.bmp', out_img[:, :, 2])
            #print(np.shape(mask))
        
    
    
    
def main():
    args = arg_parser()
    model = UNet(args)
    
    
    if args.mode == 'train':
        model.load_train_data()
        model.build_network()
        model.train()
    elif args.mode == 'test':
        model.load_test_data()
        model.build_network()
        model.test()
    
    

if __name__  == '__main__':
    main()
    