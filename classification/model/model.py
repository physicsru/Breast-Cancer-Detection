import torch
import torch.nn as nn
import torch.optim as optim


class clf_network(nn.Module):

    def __init__(self):
        super(clf_network, self).__init__()

        self.dense1 = nn.Linear(2048, 1024)
        self.act1 = nn.ReLU()
        self.norm1 = nn.BatchNorm1d(1024)

        self.dense2 = nn.Linear(1024, 512)
        self.act2 = nn.ReLU()
        self.norm2 = nn.BatchNorm1d(512)

        self.dense3 = nn.Linear(512,256)
        self.act3 = nn.ReLU()
        self.norm3 = nn.BatchNorm1d(256)

        self.dropout = nn.Dropout(p=0.3)
        self.dense4 = nn.Linear(256, 2)
        self.softmax = nn.Softmax(dim=1)


    def forward(self, input_tensor):
        dense1 = self.dense1(input_tensor)
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
        #softmax = self.softmax(dense4)

        return dense4