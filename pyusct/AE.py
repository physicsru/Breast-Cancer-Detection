import torch
import torchvision
from torch import nn
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader
import numpy as np
# TODO cycle padding
class Autoencoder(nn.Module):
    def __init__(self):
        super(Autoencoder, self).__init__()

        self.encoder_conv = nn.Sequential(
            nn.Conv2d(16, 8, kernel_size=(3, 3), padding=(1, 1)),
            nn.BatchNorm2d(8),
            nn.ReLU(True), 
        )
        self.encoder_conv2 = nn.Sequential(
            nn.Conv2d(8, 4, kernel_size=(3, 3), padding=(1, 1)),
            nn.BatchNorm2d(4),
            nn.ReLU(True), 
        )
        
        self.pool = nn.MaxPool2d(kernel_size=(3, 3), return_indices=True)
        self.encoder_linear = nn.Sequential(
            nn.Linear(22440, 800),
            nn.ReLU(True),
        )
        
        self.decoder_linear = nn.Sequential(
            nn.Linear(800, 22440),
            nn.ReLU(True),
        )
        self.unpool = nn.MaxUnpool2d(kernel_size=(3, 3))
        
        self.decoder_deconv = nn.Sequential(
            nn.ConvTranspose2d(8, 16, kernel_size=(3, 3), padding=(1, 1)),
            nn.BatchNorm2d(16),
            nn.ReLU(True),
        )
        self.decoder_deconv2 = nn.Sequential(
            nn.ConvTranspose2d(4, 8, kernel_size=(3, 3), padding=(1, 1)),
            nn.BatchNorm2d(8),
            nn.ReLU(True),
        )

    def forward(self, x):
        # encoder
        encoded, indices, shape, shaped_pooled = self.encode(x)
        # decoder
        decoded = self.decode(encoded, indices, shape, shaped_pooled)
        return encoded, decoded
    
    def encode(self, x):
        x = self.encoder_conv(x)
        x = self.encoder_conv2(x)
        shape = x.shape
        x, indices = self.pool(x)
        shape_pooled = x.shape
        x = x.view(shape[0], -1)
        # print(shape, shape_pooled, x.shape)
        x = self.encoder_linear(x)
        return x, indices, shape, shape_pooled
        
    
    def decode(self, x, indices, shape, shape_pooled):
        x = self.decoder_linear(x)
        x = x.view(shape_pooled)
        x = self.unpool(x, indices, shape)
        x = self.decoder_deconv2(x)
        x = self.decoder_deconv(x)
        return x
    