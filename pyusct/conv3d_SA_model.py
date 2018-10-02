import torch
import torchvision
from torch import nn
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader
import numpy as np
import torch.nn as nn
import torch.optim as optim
# TODO cycle padding
class Clf_conv3d_SA(nn.Module):
    def __init__(self):
        super(Clf_conv3d_SA, self).__init__()

        self.encoder_conv = nn.Sequential(
            nn.Conv3d(1, 16, kernel_size=(3, 3, 3), padding=(1, 1, 1)),
            nn.BatchNorm3d(16),
            nn.ReLU(True), 
        )
        self.encoder_conv2 = nn.Sequential(
            nn.Conv3d(16, 16, kernel_size=(3, 3, 3), padding=(1, 1, 1)),
            nn.BatchNorm3d(16),
            nn.ReLU(True), 
        )
        
        self.pool1 = nn.MaxPool3d(kernel_size=(3, 3, 3), return_indices=True)
        self.encoder_conv3 = nn.Sequential(
            nn.Conv3d(16, 32, kernel_size=(3, 3, 3), padding=(1, 1, 1)),
            nn.BatchNorm3d(32),
            nn.ReLU(True), 
        )
        self.encoder_conv4 = nn.Sequential(
            nn.Conv3d(32, 32, kernel_size=(3, 3, 3), padding=(1, 1, 1)),
            nn.BatchNorm3d(32),
            nn.ReLU(True), 
        )
        
        self.pool2 = nn.MaxPool3d(kernel_size=(3, 3, 3), return_indices=True)
        
        self.encoder_conv5 = nn.Sequential(
            nn.Conv3d(32, 64, kernel_size=(3, 3, 3), padding=(1, 1, 1)),
            nn.BatchNorm3d(64),
            nn.ReLU(True), 
        )
        self.encoder_conv6 = nn.Sequential(
            nn.Conv3d(64, 64, kernel_size=(3, 3, 3), padding=(1, 1, 1)),
            nn.BatchNorm3d(64),
            nn.ReLU(True), 
        )
        self.encoder_linear = nn.Sequential(
            nn.Linear(39424, 800),
            nn.ReLU(True),
        )
        
        self.dense1 = nn.Linear(800, 512)
        self.act1 = nn.ReLU()
        self.norm1 = nn.BatchNorm1d(512)

        self.dense2 = nn.Linear(512, 256)
        self.act2 = nn.ReLU()
        self.norm2 = nn.BatchNorm1d(256)

        self.dense3 = nn.Linear(256, 128)
        self.act3 = nn.ReLU()
        self.norm3 = nn.BatchNorm1d(128)

        self.dropout = nn.Dropout(p=0.3)
        self.dense4 = nn.Linear(128, 1)
        # self.softmax = nn.Softmax(dim=1)
        

    def forward(self, x):
        # encoder
        encoded = self.encode(x)
        dense1 = self.dense1(encoded)
        act1 = self.act1(dense1)
        norm1 = self.norm1(act1)

        dense2 = self.dense2(norm1)
        act2 = self.act2(dense2)
        norm2 = self.norm2(act2)

        dense3 = self.dense3(norm2)
        act3 = self.act3(dense3)
        norm3 = self.norm3(act3)

        dropout = self.dropout(norm3)
        dense4 = self.dense4(dropout)
        return dense4
    
    def encode(self, x):
        x = self.encoder_conv(x)
        x = self.encoder_conv2(x)
        shape = x.shape
        x, indices = self.pool1(x)
        shape_pooled = x.shape
        
        x = self.encoder_conv3(x)
        x = self.encoder_conv4(x)
        shape = x.shape
        x, indices = self.pool2(x)
        shape_pooled = x.shape
        x = self.encoder_conv5(x)
        x = self.encoder_conv6(x)
        shape = x.shape
        x = x.view(shape[0], -1)
        # print(shape, shape_pooled, x.shape)
        x = self.encoder_linear(x)
        return x



