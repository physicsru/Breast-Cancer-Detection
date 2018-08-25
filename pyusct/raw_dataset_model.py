import os, sys, math, pickle
sys.path.append("../preprocess/pyusct/")

# from pyusct/
from scaler import RFScaler
from AE import Autoencoder
# from model/
from clf_model import clf_network

import torch
from torch import nn


class Raw_dataset_clf():
    def __init__(self, AE_weight_path, scaler_path, model_params=None):
        
        self.scaler = pickle.load(open(scaler_path, 'rb'))
        
        self.ae_network = Autoencoder().cuda()
        self.clf_network = clf_network().cuda()
        
        if model_params:
            self.ae_network.load_state_dict(torch.load(model_params[0]))
            self.clf_network.load_state_dict(torch.load(model_params[1]))
        else:
            #self.ae_network.load_state_dict(torch.load(AE_weight_path))
            pass
            
        
        return
    
    def get_params(self):
        return list(self.ae_network.parameters()) + list(self.clf_network.parameters())
        
    # input: Variable
    def pred(self, data):
        compressed_data, _ = self.ae_network(data)
        pred = self.clf_network(compressed_data)
        return pred
    
    def save_model(self, output_path, name):
        torch.save(self.ae_network.state_dict(), output_path + 'ae_' + name)
        torch.save(self.clf_network.state_dict(), output_path + 'clf_' + name)
        return
    
    def reload_params(self, model_params):
        self.ae_network.load_state_dict(torch.load(model_params[0]))
        self.clf_network.load_state_dict(torch.load(model_params[1]))
        return
    
    def __str__(self):
        return "2d"
        


