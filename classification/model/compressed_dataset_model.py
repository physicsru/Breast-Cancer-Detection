import os, sys, glob, time, math, pickle
sys.path.append("../preprocess/pyusct/")
sys.path.append("./model/")
from rfdata import RFdata
from AE import Autoencoder, RFCompressedDataset

import numpy as np
from sklearn import metrics
from sklearn.model_selection import train_test_split

import torch
from torch import nn
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader

from model import clf_network


class compressed_dataset_clf():
    
    def __init__(self, dataset_dir, lr=1e-3, epochs=100000, batch_size=32, random_state=42):
        self.clf_network = clf_network().cuda()
        
        self.lr, self.epochs, self.batch_size = lr, epochs, batch_size
        self.input_list = sorted(glob.glob(os.path.join(dataset_dir, "input/*.npy")))
        self.output_list = sorted(glob.glob(os.path.join(dataset_dir, "output/*.npy")))
        X, y = [], []
        for input_file, output_file in zip(input_list, output_list):
            X.append(np.load(input_file))
            y.append(np.load(output_file))
        self.X, self.y = np.concatenate(X, axis=0), np.concatenate(y, axis=0)
        X_train, X_test, y_train, y_test = train_test_split(self.X, self.y, test_size=0.2, random_state=42)
        
        self.traindataset = RFCompressedDataset(X_train, y_train)
        self.testdataset = RFCompressedDataset(X_test, y_test)
        return 
    

    def train(self, model_output_path):
        dataloader = DataLoader(self.traindataset, self.batch_size, 
                                shuffle=True)
        
        all_params = list(self.clf_network.parameters())
        
        optimizer = torch.optim.Adam(all_params, lr=self.lr, weight_decay=1e-4)
        criterion = nn.CrossEntropyLoss().cuda()
        print('start training')
        
        for epoch in range(self.epochs):
            start_time = time.time()
            for i, data, label in enumerate(dataloader):
                data = Variable(data).cuda().float()
                label = Variable(label).view(-1).cuda().long()
                
                clf_pred = self.clf_network(data)
                clf_loss = criterion(clf_pred, label)

                optimizer.zero_grad()
                clf_loss.backward()
                optimizer.step()

                if i % 100 == 0:
                    print('Epoch:', epoch, 'Iter', i, 'Loss:', clf_loss.item(), 'Time:', time.time()-start_time)
                start_time = time.time()
            
            if epoch % 10000 == 0:
                torch.save(self.clf_network.state_dict(),  model_output_path + 'clf_compressed_data_epoch_'+str(epoch)+'.pth')
                print('Model saved')


    def test(self, model_params=None):
        dataloader = DataLoader(self.testdataset, self.batch_size, 
                                shuffle=True, num_workers=0)
        if model_params:
            self.clf_network = network.clf_network().cuda()
            self.clf_network.load_state_dict(torch.load(model_params))

        acc_list, f1_list, pre_list, recall_list = [], [], [], []
        start_time = time.time()
        for i, (data, label) in enumerate(dataloader):
            data = Variable(data).cuda().float()
            clf_pred = self.clf_network(data)
            prob_out = clf_pred.detach().cpu().numpy()
            prob_idx = np.argmax(prob_out, 1)
            
            accuracy = metrics.accuracy_score(label, prob_idx)
            precision = metrics.precision_score(label, prob_idx)
            recall = metrics.recall_score(label, prob_idx)
            f1_score = metrics.f1_score(label, prob_idx)
            acc_list.append(accuracy)
            pre_list.append(precision)
            recall_list.append(recall)
            f1_list.append(f1_score)

            print('Iter', i, 'Time:', time.time()-start_time)
            print(accuracy, f1_score, precision, recall)
            start_time = time.time()

        print('accuracy:', np.mean(acc_list))
        print('f1_score:', np.mean(f1_list))
        print('precision:', np.mean(pre_list))
        print('recall', np.mean(recall_list))
