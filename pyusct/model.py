import os, sys, math, pickle

from scaler import RFScaler
from conv3d_model import Clf_conv3d
import torch
from torch import nn

class Conv_prototype():
    
    def __init__(self, scaler_path):
        
        self.scaler = pickle.load(open(scaler_path, 'rb'))
        
    def get_params(self):
        return list(self.model.parameters())
        
    # input: Variable
    def pred(self, data):
        pred = self.model(data)
        return pred
        
    def save_model(self, output_path, type, name):
        torch.save(self.model.state_dict(), output_path + type + "_" + name)
        return
    
    def reload_params(self, model_params):
        self.model.load_state_dict(torch.load(model_params))
        return

    
class Conv_3d(Conv_prototype):
    
    def __init__(self, scaler_path, model_params=None):
        
        Conv_prototype.__init__(self, scaler_path)
        
        self.model = Clf_conv3d().cuda()
        self.model = nn.DataParallel(self.model)
        return

    
    def __str__(self):
        return "3d"

        