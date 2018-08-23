import os, sys, glob, time, math
sys.path.append("../pyusct/")

# from model/
from raw_dataset_model_3d import Raw_dataset_clf
# from pyusct/
from pytorch_dataset import RFFullDataset3d

import numpy as np
from sklearn import metrics
from sklearn.model_selection import train_test_split

import torch
from torch import nn
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader

import matplotlib.pyplot as plt

class Raw_dataset_trainer():
    def __init__(self, dataset_dir, AE_weight_path, scaler_path, model_output_path, lr=1e-3, epochs=100, batch_size=32, random_state=42):
        dataset_name = dataset_dir.split('/')[-2]
        self.output_path = os.path.join(model_output_path, dataset_name + '_lr_' + str(lr) + '_epoch_' + str(epochs) + '_batchsize_' + str(batch_size) + '/')
        if not os.path.exists(self.output_path):
            os.makedirs(self.output_path)
        self.model = Raw_dataset_clf(AE_weight_path, scaler_path)
        
        self.lr, self.epochs, self.batch_size = lr, epochs, batch_size
        self.input_list = sorted(glob.glob(os.path.join(dataset_dir, "input/*.npy")))
        self.output_list = sorted(glob.glob(os.path.join(dataset_dir, "output/*.npy")))
        X_train, X_test, y_train, y_test = train_test_split(self.input_list, self.output_list, test_size=0.2, random_state=42)
        self.traindataset = RFFullDataset3d(X_train, y_train, self.model.scaler)
        self.testdataset = RFFullDataset3d(X_test, y_test, self.model.scaler)
        self.train_loss = []
        self.valid_loss = []
        return
        

    def train(self):
        dataloader_train = DataLoader(self.traindataset, self.batch_size, 
                                shuffle=True)
        
        dataloader_valid = DataLoader(self.testdataset, self.batch_size, 
                                shuffle=True)
        all_params = self.model.get_params()
        optimizer = torch.optim.Adam(all_params, lr=self.lr, weight_decay=1e-4)
        criterion = nn.CrossEntropyLoss().cuda()
        print('start training')
        
        for epoch in range(self.epochs):
            start_time = time.time()
            train_loss = []
            valid_loss = []
            
            # train
            for i, (data, label) in enumerate(dataloader_train):
                data = Variable(data).cuda().float()
                label = Variable(label).view(-1).cuda().long()
                
                pred = self.model.pred(data)
                clf_loss = criterion(pred, label)
                train_loss.append(clf_loss.item())
                optimizer.zero_grad()
                clf_loss.backward()
                optimizer.step()

                #if i+1 % 10 == 0:
                print('Epoch:', epoch, 'Iter', i, 'Loss:', clf_loss.item(), 'Time:', time.time()-start_time)
                start_time = time.time()
                
            # valid
            for i, (data, label) in enumerate(dataloader_valid):
                data = Variable(data).cuda().float()
                label = Variable(label).view(-1).cuda().long()
                
                pred = self.model.pred(data)
                clf_loss = criterion(pred, label)
                valid_loss.append(clf_loss.item())
                if i > 3000: break
            
            self.train_loss.append(np.mean(train_loss))
            self.valid_loss.append(np.mean(valid_loss))
            if (epoch+1) % 10 == 0:
                name = 'raw_data_epoch_'+str(epoch)+'.pth'
                self.model.save_model(self.output_path, name)
                print('Model saved')
        return


    def test(self, model_params=None):
        dataloader = DataLoader(self.testdataset, self.batch_size, 
                                shuffle=True)
        if model_params:
            self.model.reload_params(model_params)
        criterion = nn.CrossEntropyLoss().cuda()
        acc_list, f1_list, pre_list, recall_list,loss_list = [], [], [], [], []
        start_time = time.time()
        for i, (data, label) in enumerate(dataloader):
            data = Variable(data).cuda().float()
            label_tensor = Variable(label).view(-1).cuda().long()
            prob_out = self.model.pred(data).detach().cpu().numpy()
            prob_idx = np.argmax(prob_out, 1)
            pred = self.model.pred(data)
            
            accuracy = metrics.accuracy_score(label, prob_idx)
            precision = metrics.precision_score(label, prob_idx)
            recall = metrics.recall_score(label, prob_idx)
            f1_score = metrics.f1_score(label, prob_idx)
            clf_loss = criterion(pred, label_tensor)
            loss_list.append(clf_loss.item())
            acc_list.append(accuracy)
            pre_list.append(precision)
            recall_list.append(recall)
            f1_list.append(f1_score)

            print('Iter', i, 'Time:', f'{(time.time()-start_time):.4f}')
            print(f'{accuracy:.4f}', f'{f1_score:.4f}', f'{precision:.4f}', f'{recall:.4f}')
            start_time = time.time()

        print('accuracy:', round(np.mean(acc_list),4))
        print('f1_score:', round(np.mean(f1_list),4))
        print('precision:', round(np.mean(pre_list),4))
        print('recall', round(np.mean(recall_list),4))
        print('loss', round(np.mean(loss_list),4))
        return
        
    def plot_learn_curve(self):
        plt.plot(self.epoch_loss)
        plt.plot(self.valid_loss)
        plt.show()
        return
        
        


