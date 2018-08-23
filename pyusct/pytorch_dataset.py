# input raw 16*256*200
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
import numpy as np

class RFFullDataset(Dataset):
    def __init__(self, input_files, output_files, scaler):
        self.file_paths = input_files
        self.label_paths = output_files
        self.scaler = scaler
        return 
        
    def __len__(self):
        return len(self.file_paths)
    
    def __getitem__(self, idx):
        data = np.load(self.file_paths[idx])
        label = np.load(self.label_paths[idx])
        shape = data.shape
        scaled_data = self.scaler.transform(data.reshape(1, -1))
        return scaled_data.reshape(shape[1],shape[2],shape[3]), label
    
class RFFullDataset3d(Dataset):
    def __init__(self, input_files, output_files, scaler):
        self.file_paths = input_files
        self.label_paths = output_files
        self.scaler = scaler
        return 
        
    def __len__(self):
        return len(self.file_paths)
    
    def __getitem__(self, idx):
        data = np.load(self.file_paths[idx])
        label = np.load(self.label_paths[idx])
        shape = data.shape
        scaled_data = self.scaler.transform(data.reshape(1, -1))
        return scaled_data.reshape(1, shape[1],shape[2],shape[3]), label

    
# input 800 dim
class RFCompressedDataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y
        return 
        
    def __len__(self):
        return len(self.y)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]