import torch
import torch.nn as nn
import torch.optim as optim



class clf_network(nn.Module):

    def __init__(self):
        super(clf_network, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(800, 160),
            nn.BatchNorm1d(160),
            nn.ReLU(),
            nn.Linear(160, 160),
            nn.BatchNorm1d(160),
            nn.ReLU(),
            nn.Linear(160, 160),
            nn.BatchNorm1d(160),
            nn.ReLU(),
            nn.Linear(160, 2),
            nn.Softmax(dim=1)
            )
        '''
        self.dense1 = nn.Linear(800, 160)
        self.act1 = nn.ReLU()
        self.norm1 = nn.BatchNorm1d(160)

        self.dense2 = nn.Linear(160, 160)
        self.act2 = nn.ReLU()
        self.norm2 = nn.BatchNorm1d(160)

        self.dense3 = nn.Linear(160, 160)
        self.act3 = nn.ReLU()
        self.norm3 = nn.BatchNorm1d(160)

        self.dropout = nn.Dropout(p=0.3)
        self.dense4 = nn.Linear(160, 2)
        self.softmax = nn.Softmax(dim=1)
        '''

    def forward(self, input_tensor):
        '''
        dense1 = self.dense1(input_tensor)
        act1 = self.act1(dense1)
        norm1 = self.norm1(act1)

        dense2 = self.dense2(norm1)
        act2 = self.act1(dense2)
        norm2 = self.norm1(act2)

        dense3 = self.dense2(norm2)
        act3 = self.act1(dense3)
        norm3 = self.norm1(act3)

        dropout = self.dropout(norm3)
        dense4 = self.dense4(dropout)
        softmax = self.softmax(dense4)
        '''
        output = self.network(input_tensor)
        return output


class cnn_network(nn.Module):

    def __init__(self):
        super(cnn_network, self).__init__()

        self.conv1 = nn.Conv2d(1, 32, (5, 5), padding=(2, 2))
        self.conv_act1 = nn.ReLU()
        self.conv_norm1 = nn.BatchNorm2d(32)

        self.conv2 = nn.Conv2d(32, 32, (5, 5), padding=(2, 2))
        self.conv_act2 = nn.ReLU()
        self.conv_norm2 = nn.BatchNorm2d(32)

        self.conv3 = nn.Conv2d(32, 1, (5, 5), padding=(2, 2))


    def forward(self, input_tensor, img_size):
        argmax = torch.argmax(input_tensor, dim=1).float()
        reshape = argmax.view(-1, 1, img_size, img_size)

        conv1 = self.conv1(reshape)
        conv_act1 = self.conv_act1(conv1)
        conv_norm1 = self.conv_norm1(conv_act1)

        conv2 = self.conv2(conv_norm1)
        conv_act2 = self.conv_act2(conv2)
        conv_norm2 = self.conv_norm1(conv_act2)

        conv3 = self.conv3(conv_norm2)

        return conv3




