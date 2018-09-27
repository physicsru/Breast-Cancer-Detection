import numpy as np
import os, json, glob, sys, time

from rfdata import RFdata
from scaler import RFScaler

import torch
from torch import nn
from torch.autograd import Variable
from AE import Autoencoder

from conv2d_model import Clf_conv2d
from conv3d_model import Clf_conv3d
from conv3d_VGG_model import Clf_conv3d_VGG


import matplotlib.pyplot as plt

class Visualizer(object):
    
    # type: "AE" "AE_fixed" "PCA"
    def __init__(self, source=None, model=None, type="AE"):
        
        self.rf = source
        self.model = model
        return 
    
    def extract_subimage(self, offset=[256, 256], shape=[100, 100], batch_size=32, type="normal"):
        res = []
        truth = np.zeros(shape)
        self.indices = generate_indices(shape, offset)
        batch = []
        for i, (ix, iy) in enumerate(self.indices):
            raw = dimension_reduce_rf_point(self.rf, ix, iy, type)
            scaled = self.model.scaler.transform(raw.reshape(1, -1))
            batch.append(scaled)
            if i % batch_size + 1 == batch_size or i == len(self.indices)-1:
                if "2d" in str(self.model):
                    data = np.array(batch).reshape((-1, 16, 256, 200))
                elif "3d" in str(self.model):
                    data = np.array(batch).reshape((-1, 1, 16, 256, 200))
                    
                data = Variable(torch.from_numpy(data)).cuda().float()
                pred = self.model.pred(data).detach().cpu().numpy()
                
                if pred.shape[1]==1:
                    # for SA pretrain
                    res.append(pred[:, 0])
                else:
                    # for clas
                    res.append(pred[:, 1])
                print("predict point {}".format(i))
                batch = []
            
            truth[ix-offset[0], iy-offset[1]] = self.rf.medium_sct[ix, iy]

        self.res = np.concatenate(res).reshape(shape)
        self.truth = truth
                
    def visualize_prob(self):
        # truth and pred
        fig = plt.figure(figsize=(16,12))
        
        # truth
        ax = plt.subplot(121)
        image = ax.imshow(self.truth, cmap='gray')
        ax.axis("image")
        plt.title("ground truth")
        # predicted
        ax = plt.subplot(122)
        image = ax.imshow(self.res, cmap='gray')
        ax.axis("image")
        plt.title("predicted")
        
        plt.show()
        return 
    
    def visualize_01(self):
        # truth and pred
        fig = plt.figure(figsize=(16,12))
        
        # truth
        ax = plt.subplot(121)
        image = ax.imshow(self.truth, cmap='gray')
        ax.axis("image")
        plt.title("ground truth")
        # predicted
        ax = plt.subplot(122)
        res = self.res
        res[res >= 0.5] = 1
        res[res < 0.5] = 0
        image = ax.imshow(res, cmap='gray')
        ax.axis("image")
        plt.title("predicted")
        
        plt.show()
        return 
        
    def test(self):
        self.res = np.zeros([100, 100])
        self.truth = np.zeros([100, 100])
        for i in range(100):
            self.res[i, 30:50] = 1
        
        self.visualize_subimage()
        return 
        
        
        
def generate_indices(shape, offset):
    indices = np.indices((shape[0], shape[1]))
    indices[0] += offset[0]
    indices[1] += offset[1]
    indices = indices.transpose(1,2,0)
    indices = indices.reshape(-1, 2)    
        
    return indices        
                
                
def dimension_reduce_rf_point(rf, ix, iy, type="normal"):
    offsets = np.arange(-100, 100)
    # Attention! matlab meshgrid -> transpose of python grid
    _, subset = rf.getPointSubset((iy, ix), offsets)
    #Attention only for the wire map! : return subset[::2, :, :]
    if type=="normal": return subset
    if type=="wire": return subset[::2, :, :]