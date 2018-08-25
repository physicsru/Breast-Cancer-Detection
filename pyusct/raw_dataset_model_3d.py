import os, sys, math, pickle
sys.path.append("../preprocess/pyusct/")

# from pyusct/
from scaler import RFScaler
from con3d_model import Clf_conv3d

import torch
from torch import nn


class Raw_dataset_clf():
    def __init__(self, AE_weight_path, scaler_path, model_params=None):
        
        self.scaler = pickle.load(open(scaler_path, 'rb'))
        
        self.model = Clf_conv3d().cuda()
        
        if model_params:
            self.model.load_state_dict(torch.load(model_params))
        
        return
    
    def get_params(self):
        return list(self.model.parameters())
        
    # input: Variable
    def pred(self, data):
        pred = self.model(data)
        return pred
    
    def save_model(self, output_path, name):
        torch.save(self.model.state_dict(), output_path + 'conv3d_' + name)
        return
    
    def reload_params(self, model_params):
        self.model.load_state_dict(torch.load(model_params))
        return
        
    def __str__(self):
        return "3d"

