import numpy as np
import os, json, glob, sys, time

from rfdata import RFdata
from scaler import RFScaler

import torch
from torch import nn
from torch.autograd import Variable
from AE import Autoencoder

from raw_dataset_model import Raw_dataset_clf


import matplotlib.pyplot as plt

class Visualizer(object):
    
    # type: "AE" "AE_fixed" "PCA"
    def __init__(self, source, scaler_path=None, model_params=None, type="AE"):
        
        self.rf = RFdata(source)
        self.model = Raw_dataset_clf(None, scaler_path, model_params)
        
        return 
    
    def extract_subimage(self, offset=[256, 256], shape=[100, 100]):
        res = []
        truth = np.zeros(shape)
        self.indices = generate_indices(shape, offset)
        batch_size = 32
        batch = []
        for i, (ix, iy) in enumerate(self.indices):
            raw = dimension_reduce_rf_point(self.rf, ix, iy)
            scaled = self.model.scaler.transform(raw.reshape(1, -1))
            batch.append(scaled)
            if i % batch_size + 1 == batch_size or i == len(self.indices)-1:
                data = np.array(batch).reshape((-1, 16, 256, 200))
                data = Variable(torch.from_numpy(data)).cuda().float()
                pred = self.model.pred(data).detach().cpu().numpy()
                res.append(pred[:,1])
                print("predict point {}".format(i))
                batch = []
            
            truth[ix-offset[0], iy-offset[1]] = self.rf.medium_sct[ix, iy]
              
        self.res = np.transpose(np.concatenate(res).reshape(shape))
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
                
                
def dimension_reduce_rf_point(rf, ix, iy):
    offsets = np.arange(-100, 100)
    _, subset = rf.getPointSubset((ix,iy), offsets)
    #Attention only for the wire map! : return subset[::2, :, :]
    return subset[:, :, :]