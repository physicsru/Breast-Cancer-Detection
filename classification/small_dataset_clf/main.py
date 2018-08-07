import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms

import numpy as np
import os, time
import model
import utils
import cv2
import argparse
from sklearn import metrics
from skimage import exposure



def arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", default = 4096, type = int)
    parser.add_argument("--clf_iter", default = 50000, type = int)
    parser.add_argument("--cnn_iter", default = 20000, type = int)
    parser.add_argument("--init_lr", default = 1e-4, type = float)
    parser.add_argument('--weight-decay', default = 5e-3, type=float)
    parser.add_argument("--save_dir", default = 'saved_models')
    parser.add_argument("--out_dir", default = 'output')
    parser.add_argument("--mode", default = 'train')

    args = parser.parse_args()
    return args


class Model():
    def __init__(self, args):
        self.batch_size = args.batch_size
        self.clf_iter = args.clf_iter
        self.cnn_iter = args.cnn_iter
        self.weight_decay = args.weight_decay
        self.save_dir = args.save_dir
        self.out_dir = args.out_dir

        self.clf_network = model.clf_network().cuda()
        self.clf_loss = nn.CrossEntropyLoss().cuda()
        self.clf_optim = optim.Adam(self.clf_network.parameters(), lr=args.init_lr , 
                                    weight_decay = self.weight_decay, betas = (0.9, 0.999), eps=1e-08)

        
    def next_batch(self, data, label, batch_size):
        idx = np.arange(0 , len(data))
        np.random.shuffle(idx)
        idx = idx[:batch_size]
        batch_data = [data[i] for i in idx]
        batch_label = [label[i] for i in idx]
    
        return np.asarray(batch_data), np.asarray(batch_label)

    def load_data(self):
        self.train_data = np.load('dataset034/data.npy')[:45000]
        self.train_label = np.load('dataset034/label.npy')[:45000]
        self.test_data = np.load('dataset034/data.npy')[45000:]
        self.test_label = np.load('dataset034/label.npy')[45000:]
        
    
    def train_clf(self):
        if not os.path.exists(self.save_dir):
            os.mkdir(self.save_dir)
     
        start = time.time()
        for i in range(self.clf_iter): 

            batch_data, batch_clf_label = self.next_batch(self.train_data, 
                                                self.train_label, self.batch_size)
            batch_data = torch.from_numpy(batch_data).float().cuda()
            batch_clf_label = torch.from_numpy(batch_clf_label).long().cuda()

            prob_out = self.clf_network(batch_data)
            clf_loss = self.clf_loss(prob_out, batch_clf_label)

            self.clf_optim.zero_grad()   
            clf_loss.backward()   
            self.clf_optim.step()

            if np.mod(i+1, 100) == 0:
                print('iter:', i+1, 'loss:', clf_loss.item(), 'time:', time.time()-start)
                start = time.time()
        
                if np.mod(i+1, 10000) == 0:
                    save_clf_dir = self.save_dir + "/clf_network_iter_{}.pth".format(i+1)
                    torch.save(self.clf_network.state_dict(), save_clf_dir)


    def train_cnn(self):

        start = time.time()
        self.clf_network.load_state_dict(torch.load(self.save_dir + 
                                "/clf_network_iter_{}.pth".format(self.clf_iter)))

        for i in range(self.cnn_iter):

            batch_data, batch_cnn_label = utils.next_cnn_batch(self.train_data, self.train_label, 
                                                                self.batch_size, self.train_img_size)
            batch_data = torch.from_numpy(batch_data).float().cuda()
            batch_cnn_label = torch.from_numpy(batch_cnn_label).float().cuda()

            prob_out = self.clf_network(batch_data)
            #argmax = torch.argmax(prob_out).float().cuda()
            cnn_out = self.cnn_network(prob_out, self.train_img_size)
            cnn_loss = self.cnn_loss(cnn_out, batch_cnn_label)

            self.cnn_optim.zero_grad()   
            cnn_loss.backward()   
            self.cnn_optim.step()

            if np.mod(i+1, 10) == 0:
                print('iter:', i+1, 'loss:', cnn_loss.item(), 'time:', time.time()-start)
                start = time.time()
        
                if np.mod(i+1, 10000) == 0:
                    save_clf_dir = self.save_dir + "/cnn_network_iter_{}.pth".format(i+1)
                    torch.save(self.cnn_network.state_dict(), save_clf_dir)
            
        

    def test(self):
        
        self.clf_network.load_state_dict(torch.load(self.save_dir + 
                                "/clf_network_iter_{}.pth".format(self.clf_iter)))
        
        test_input = torch.from_numpy(self.test_data).float().cuda()
        prob_out = self.clf_network(test_input)
        prob_out = prob_out.detach().cpu().numpy()
        prob_idx = np.argmax(prob_out, 1)

        accuracy = metrics.accuracy_score(self.test_label, prob_idx)
        f1_score = metrics.f1_score(self.test_label, prob_idx)
        precision = metrics.precision_score(self.test_label, prob_idx)
        recall = metrics.recall_score(self.test_label, prob_idx)
        print('accuracy:', accuracy)
        print('f1_score:', f1_score)
        print('precision:', precision)
        print('recall', recall)

        
        prob_out = np.reshape(prob_out[:, :1], (512, 512))
        prob_idx = np.reshape(prob_idx, (512, 512))
        cv2.imwrite('argmax.bmp', prob_idx*255)
        cv2.imwrite('prob.bmp', prob_out*255)
        cv2.imwrite('label.bmp', self.test_label.reshape(512, 512)*255)
        
    
def main():
    args = arg_parser()
    model = Model(args)
    
    
    if args.mode == 'train':
        
        model.load_data()
        model.train_clf()
        #model.train_cnn()
        
    elif args.mode == 'test':
        
        model.load_data()
        model.test()
    
    

if __name__  == '__main__':
    main()
    