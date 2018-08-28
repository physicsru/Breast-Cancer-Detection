import torch
import torchvision
from torch import nn
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader
import numpy as np
import torch.nn as nn
import torch.optim as optim
# TODO cycle padding
class Clf_conv3d(nn.Module):
    def __init__(self):
        super(Clf_conv3d, self).__init__()

        self.encoder_conv1_1 = nn.Sequential(
            nn.Conv3d(1, 32, kernel_size=(3, 3, 3), padding=(2, 2, 2), stride=1),
            nn.BatchNorm3d(32),
            nn.ReLU(True), 
        )
        self.encoder_conv1_2 = nn.Sequential(
            nn.Conv3d(32, 32, kernel_size=(3, 3, 3), padding=(2, 2, 2), stride=1),
            nn.BatchNorm3d(32),
            nn.ReLU(True), 
        )
        
        self.pool1 = nn.MaxPool3d(kernel_size=(3, 3, 3), return_indices=True)
        
        self.encoder_conv2_1 = nn.Sequential(
            nn.Conv3d(32, 64, kernel_size=(3, 3, 3), padding=(2, 2, 2), stride=1),
            nn.BatchNorm3d(64),
            nn.ReLU(True), 
        )
        self.encoder_conv2_2 = nn.Sequential(
            nn.Conv3d(64, 64, kernel_size=(3, 3, 3), padding=(2, 2, 2), stride=1),
            nn.BatchNorm3d(64),
            nn.ReLU(True), 
        )
        
        self.pool2 = nn.MaxPool3d(kernel_size=(3, 3, 3), return_indices=True)  
        
        self.encoder_conv3_1 = nn.Sequential(
            nn.Conv3d(64, 128, kernel_size=(3, 3, 3), padding=(2, 2, 2), stride=1),
            nn.BatchNorm3d(128),
            nn.ReLU(True), 
        )
        self.encoder_conv3_2 = nn.Sequential(
            nn.Conv3d(128, 128, kernel_size=(3, 3, 3), padding=(2, 2, 2), stride=1),
            nn.BatchNorm3d(128),
            nn.ReLU(True), 
        )
        self.encoder_conv3_3 = nn.Sequential(
            nn.Conv3d(128, 128, kernel_size=(3, 3, 3), padding=(2, 2, 2), stride=1),
            nn.BatchNorm3d(128),
            nn.ReLU(True), 
        )
        
        self.pool3 = nn.MaxPool3d(kernel_size=(3, 3, 3), return_indices=True)
        
        self.encoder_conv4_1 = nn.Sequential(
            nn.Conv3d(128, 256, kernel_size=(3, 3, 3), padding=(2, 2, 2), stride=1),
            nn.BatchNorm3d(256),
            nn.ReLU(True), 
        )
        self.encoder_conv4_2 = nn.Sequential(
            nn.Conv3d(256, 256, kernel_size=(3, 3, 3), padding=(2, 2, 2), stride=1),
            nn.BatchNorm3d(256),
            nn.ReLU(True), 
        )
        self.encoder_conv4_3 = nn.Sequential(
            nn.Conv3d(256, 256, kernel_size=(3, 3, 3), padding=(2, 2, 2), stride=1),
            nn.BatchNorm3d(256),
            nn.ReLU(True), 
        )
        
        self.pool4 = nn.MaxPool3d(kernel_size=(3, 3, 3), return_indices=True)
        
        self.encoder_conv5_1 = nn.Sequential(
            nn.Conv3d(256, 256, kernel_size=(3, 3, 3), padding=(2, 2, 2), stride=1),
            nn.BatchNorm3d(256),
            nn.ReLU(True), 
        )
        self.encoder_conv5_2 = nn.Sequential(
            nn.Conv3d(256, 256, kernel_size=(3, 3, 3), padding=(2, 2, 2), stride=1),
            nn.BatchNorm3d(256),
            nn.ReLU(True), 
        )
        self.encoder_conv5_3 = nn.Sequential(
            nn.Conv3d(256, 256, kernel_size=(3, 3, 3), padding=(2, 2, 2), stride=1),
            nn.BatchNorm3d(256),
            nn.ReLU(True), 
        )
        
        self.pool5 = nn.MaxPool3d(kernel_size=(3, 3, 3), return_indices=True)
        
        self.encoder_linear = nn.Sequential(
            nn.Linear(9216, 800),
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
        self.dense4 = nn.Linear(128, 2)
        self.softmax = nn.Softmax(dim=1)
        

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
        x = self.encoder_conv1_1(x)
        x = self.encoder_conv1_2(x)
        shape = x.shape
        x, indices = self.pool1(x)
        shape_pooled = x.shape
        x = self.encoder_conv2_1(x)
        x = self.encoder_conv2_2(x)
        shape = x.shape
        x, indices = self.pool2(x)
        shape_pooled = x.shape
        x = self.encoder_conv3_1(x)
        x = self.encoder_conv3_2(x)
        x = self.encoder_conv3_3(x)
        shape = x.shape
        x, indices = self.pool3(x)
        shape_pooled = x.shape
        x = self.encoder_conv4_1(x)
        x = self.encoder_conv4_2(x)
        x = self.encoder_conv4_3(x)
        shape = x.shape
        x, indices = self.pool4(x)
        shape_pooled = x.shape
        x = self.encoder_conv5_1(x)
        x = self.encoder_conv5_2(x)
        x = self.encoder_conv5_3(x)
        shape = x.shape
        batch_size = shape[0]
        x, indices = self.pool5(x)
        shape_pooled = x.shape      
        x = x.view(batch_size, -1)
        # print(shape, shape_pooled, x.shape)
        x = self.encoder_linear(x)
        return x



