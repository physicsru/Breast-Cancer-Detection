import os, sys, glob, time, math, pickle
sys.path.append("../preprocess/pyusct/")

# from pyusct/
from AE import Autoencoder
# from model/
from model import clf_network

import torch
from torch import nn


class Compressed_dataset_clf():
    
    def __init__(self, model_params=None):
        self.clf_network = clf_network().cuda()
        
        if model_params:
            self.clf_network.load_state_dict(torch.load(model_params))
        return
    
    def get_params(self):
        return list(self.clf_network.parameters())

    # input: Variable
    def pred(self, data):
        pred = self.clf_network(data)
        return pred
    
    def save_model(self, output_path, name):
        torch.save(self.clf_network.state_dict(), output_path + 'clf_' + name)
        return
    
    def reload_params(self, model_params):
        self.clf_network.load_state_dict(torch.load(model_params))
        return
