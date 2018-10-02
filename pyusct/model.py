import os, sys, math, pickle

from scaler import RFScaler

from AE import Autoencoder
from conv2d_model import Clf_conv2d
from conv3d_model import Clf_conv3d
from conv3d_VGG_model import Clf_conv3d_VGG
from conv3d_VGG_SA_model import Clf_conv3d_VGG_SA
from conv3d_SA_model import Clf_conv3d_SA
from conv3d_ce_model import Clf_conv3d_ce

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

class Conv_3d_VGG_transfer(Conv_prototype):
    
    def __init__(self, scaler_path, model_params_SA=None):
        
        Conv_prototype.__init__(self, scaler_path)
        
        self.model_SA = Clf_conv3d_VGG_SA().cuda()
        self.model_ce = Clf_conv3d_ce().cuda()
        
        if model_params_SA:
            self.model_SA.load_state_dict(torch.load(model_params_SA))
        
        return
    
    def get_params(self):
        return list(self.model_SA.parameters()) + list(self.model_ce.parameters())
    
    def pred(self, data):
        pred = self.model_SA(data)
        pred = self.model_ce(pred)
        return pred
    
    def save_model(self, output_path, type, name):
        torch.save(self.model_SA.state_dict(), output_path + type + "_" + 'SA_' + name)
        torch.save(self.model_ce.state_dict(), output_path + type + "_" + 'ce_' + name)
        return
    
    def reload_params(self, model_params):
        self.model_SA.load_state_dict(torch.load(model_params[0]))
        self.model_ce.load_state_dict(torch.load(model_params[1]))
        return
    
    def __str__(self):
        return "3d_VGG_transfer"    
    
class Conv_3d_transfer(Conv_prototype):
    
    def __init__(self, scaler_path, model_params_SA=None):
        
        Conv_prototype.__init__(self, scaler_path)
        
        self.model_SA = Clf_conv3d_SA().cuda()
        self.model_ce = Clf_conv3d_ce().cuda()
        
        if model_params_SA:
            self.model_SA.load_state_dict(torch.load(model_params_SA))
        
        return
    
    def get_params(self):
        return list(self.model_SA.parameters()) + list(self.model_ce.parameters())
    
    def pred(self, data):
        pred = self.model_SA(data)
        pred = self.model_ce(pred)
        return pred
    
    def save_model(self, output_path, type, name):
        torch.save(self.model_SA.state_dict(), output_path + type + "_" + 'SA_' + name)
        torch.save(self.model_ce.state_dict(), output_path + type + "_" + 'ce_' + name)
        return
    
    def reload_params(self, model_params):
        self.model_SA.load_state_dict(torch.load(model_params[0]))
        self.model_ce.load_state_dict(torch.load(model_params[1]))
        return
    
    def __str__(self):
        return "3d_transfer"
    
class Conv_3d_SA(Conv_prototype):
    
    def __init__(self, scaler_path, model_params=None):
        
        Conv_prototype.__init__(self, scaler_path)
        
        self.model = Clf_conv3d_SA().cuda()
        
        if model_params:
            self.model.load_state_dict(torch.load(model_params))
        
        return
    
    def __str__(self):
        return "3d_SA"
    
class Conv_3d_VGG_SA(Conv_prototype):
    
    def __init__(self, scaler_path, model_params=None):
        
        Conv_prototype.__init__(self, scaler_path)
        
        self.model = Clf_conv3d_VGG_SA().cuda()
        
        if model_params:
            self.model.load_state_dict(torch.load(model_params))
        
        return
    
    def __str__(self):
        return "3d_VGG_SA"
    
class Conv_3d_VGG(Conv_prototype):
    
    def __init__(self, scaler_path, model_params=None):
        
        Conv_prototype.__init__(self, scaler_path)
        
        self.model = Clf_conv3d_VGG().cuda()
        
        if model_params:
            self.model.load_state_dict(torch.load(model_params))
        
        return
    
    def __str__(self):
        return "3d_VGG"
    
class Conv_3d_VGG_parallel(Conv_prototype):
    
    def __init__(self, scaler_path, model_params=None):
        
        Conv_prototype.__init__(self, scaler_path)
        
        self.model = Clf_conv3d_VGG().cuda()
        
        if model_params:
            self.model.load_state_dict(torch.load(model_params))
        
        self.model = nn.DataParallel(self.model)
        return
    
    def __str__(self):
        return "3d_VGG"
    
    
class Conv_3d(Conv_prototype):
    
    def __init__(self, scaler_path, model_params=None):
        
        Conv_prototype.__init__(self, scaler_path)
        
        self.model = Clf_conv3d().cuda()
        
        if model_params:
            self.model.load_state_dict(torch.load(model_params))
        
        return
    
    def __str__(self):
        return "3d"
    
    
class Conv_2d(Conv_prototype):
    
    def __init__(self, AE_weight_path, scaler_path, model_params=None):
        
        Conv_prototype.__init__(self, scaler_path)
        
        self.ae_network = Autoencoder().cuda()
        self.clf_network = Clf_conv2d().cuda()
        
        if model_params:
            self.ae_network.load_state_dict(torch.load(model_params[0]))
            self.clf_network.load_state_dict(torch.load(model_params[1]))
        else:
            # useless
            # self.ae_network.load_state_dict(torch.load(AE_weight_path))
            pass
        return
    
    def get_params(self):
        return list(self.ae_network.parameters()) + list(self.clf_network.parameters())
    
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

        