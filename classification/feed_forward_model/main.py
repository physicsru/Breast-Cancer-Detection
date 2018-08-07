import tensorflow as tf 
import numpy as np
import tensorflow.contrib.slim as slim
import os, time
import model
import utils
import cv2
import argparse
from skimage import exposure


'''
settings of the whole program, for example, you can 
use python main.py --mode=test in the terminal 
'''

def arg_parser():
    parser = argparse.ArgumentParser()
    #parser.add_argument("--img_size", default = 48, type = int)
    parser.add_argument("--batch_size", default = 16, type = int)
    parser.add_argument("--iterations", default = 50000, type = int)
    parser.add_argument("--init_lr", default = 1e-4, type = float)
    parser.add_argument("--save_dir", default = 'saved_models')
    parser.add_argument("--out_dir", default = 'output')
    parser.add_argument("--mode", default = 'test')

    args = parser.parse_args()
    return args


class Model():
    def __init__(self, args):
        self.batch_size = args.batch_size
        self.iterations = args.iterations
        self.init_lr = args.init_lr
        self.save_dir = args.save_dir
        self.out_dir = args.out_dir
    
    
    
    def build_network(self, img_size):
        self.img_size = img_size
        self.input = tf.placeholder(tf.float32, [None, 800])
        self.image_label = tf.placeholder(tf.float32, [None, self.img_size, self.img_size, 1])
        self.point_label = tf.placeholder(tf.float32, [None, 2])
        self.learning_rate = tf.placeholder(tf.float32)
        self.train_bool = tf.placeholder(tf.bool)
        self.prob_out, self.img_out = model.network_two_branchs(self.input, 
                                                img_size=self.img_size, is_training=self.train_bool)
        self.clf_loss = tf.losses.softmax_cross_entropy(self.point_label, self.prob_out)
        self.img_loss = tf.reduce_mean(tf.losses.absolute_difference(self.image_label, self.img_out))
        self.loss = self.clf_loss + 1e-3*self.img_loss
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            self.optimizer = tf.train.AdamOptimizer(self.learning_rate).minimize(self.loss)
        tf.summary.scalar("Loss", self.loss)
        self.sess = tf.Session()
        self.saver = tf.train.Saver()
        
        
    
    '''
    def load_train_data(self):
        self.train_data = utils.load_data()
        self.train_label = utils.load_label() 
    '''
        
        
    def load_test_data(self):
        self.test_data = utils.load_test_data()
        #self.test_label = utils.load_label()

    
    
    def train(self):
        if not os.path.exists(self.save_dir):
            os.mkdir(self.save_dir)
        '''
        else:
            self.saver.restore(self.sess, tf.train.latest_checkpoint(self.save_dir))
        '''
        init = tf.global_variables_initializer()
        summary_op = tf.summary.merge_all()
        train_writer = tf.summary.FileWriter(self.save_dir+"/train", self.sess.graph)
        self.sess.run(init)
        lr = self.init_lr
        start = time.time()
        
        for i in range(self.iterations):
            
            if np.mod(i, 1000) == 0:
                
                self.train_data, self.train_label = None, None
                self.train_data, self.train_label = utils.load_data_label()
                #print(np.shape(self.train_data), np.shape(self.train_label))
  
            
            data, image_label, point_label = utils.next_batch(self.train_data, self.train_label, 
                                                                self.img_size, self.batch_size)
            summary, loss, _ = self.sess.run([summary_op, self.loss, self.optimizer], 
                                             feed_dict = {self.input: data, self.image_label: image_label, 
                                                        self.point_label: point_label, self.learning_rate: lr,
                                                        self.train_bool: True})
            train_writer.add_summary(summary, i)
            
            if np.mod(i+1, 10) == 0:
                print('iter:', i+1, 'loss:', loss, 'time:', time.time()-start)
                start = time.time()
        
            if np.mod(i+1, 10000) == 0:
                self.saver.save(self.sess, self.save_dir+"/model")
            
        

    def test(self):
        if not os.path.exists(self.out_dir):
            os.mkdir(self.out_dir)
        self.saver.restore(self.sess, tf.train.latest_checkpoint(self.save_dir))
        for i in range(len(self.test_data)):
            img = np.reshape(self.test_data[i], (-1, 800))
            mask = self.sess.run([self.img_out], feed_dict = {self.input: img, self.train_bool: False})
            #print(np.shape(mask))
            mask = np.asarray(mask).reshape(512, 512, 1)
            mask = exposure.rescale_intensity(np.squeeze(mask), out_range=(0, 255))
            save_dir = os.path.join(self.out_dir, str(i).zfill(4)+'.bmp')
            
            cv2.imwrite(save_dir, mask)
            

    
    
    
def main():
    args = arg_parser()
    model = Model(args)
    
    
    if args.mode == 'train':
        
        model.build_network(64)
        #model.load_train_data()
        model.train()
        
    elif args.mode == 'test':
        
        model.build_network(512)
        model.load_test_data()
        model.test()
    
    

if __name__  == '__main__':
    main()
    