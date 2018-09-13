import torch
import torch.nn as nn
import torch.optim as optim


class Clf_conv3d_ce(nn.Module):

    def __init__(self):
        super(Clf_conv3d_ce, self).__init__()
        self.dense = nn.Linear(1, 2)


    def forward(self, input_tensor):
        dense = self.dense(input_tensor)
        
        return dense