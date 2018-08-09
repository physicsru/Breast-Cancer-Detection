import os, sys, glob, math
sys.path.append("../../preprocess/pyusct/")
from rfdata import RFdata
from scaler import RFScaler
from AE import Autoencoder, RFFullDataset
from network import clf_network

import numpy as np
from sklearn import metrics

import torch, torchvision
from torch import nn
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from scaler import RFScaler
from sklearn.model_selection import train_test_split

import pickle, time


class clf():
    def __init__(self, dataset_dir, scaler_path):
        self.lr = 1e-3
        self.epochs = 25
        self.batch_size = 32
        self.input_list = sorted(glob.glob(os.path.join(dataset_dir, "input/*.npy")))
        self.output_list = sorted(glob.glob(os.path.join(dataset_dir, "output/*.npy")))
        self.scaler = pickle.load(open(scaler_path, 'rb'))
        X_train, X_test, y_train, y_test = train_test_split(self.input_list, self.output_list, test_size=0.2, random_state=42)
        self.traindataset = RFFullDataset(X_train, y_train, self.scaler)
        self.testdataset = RFFullDataset(X_test, y_test, self.scaler)
        print(self.lr, self.epochs)
        

    def train(self, AE_weight_path, model_output_path):
        
        dataloader = DataLoader(self.traindataset, self.batch_size, 
                                shuffle=True, num_workers=0)
        
        self.ae_network = Autoencoder().cuda()
        self.clf_network = clf_network().cuda()
        
        self.ae_network.load_state_dict(torch.load(AE_weight_path))
        all_params = list(self.ae_network.parameters()) + list(self.clf_network.parameters())
        
        optimizer = torch.optim.Adam(all_params, lr=self.lr, weight_decay=1e-4)
        criterion = nn.CrossEntropyLoss().cuda()
        print('start training')
        
        for epoch in range(self.epochs):
            start_time = time.time()
            for i, data_label in enumerate(dataloader):
                data = Variable(data_label[0]).cuda().float()
                label = Variable(data_label[1]).view(-1).cuda().long()
                squeeze_data, _ = self.ae_network(data)
                
                clf_pred = self.clf_network(squeeze_data)
                clf_loss = criterion(clf_pred, label)

                optimizer.zero_grad()
                clf_loss.backward()
                optimizer.step()

                #if i+1 % 10 == 0:
                print('Epoch:', epoch, 'Iter', i, 'Loss:', clf_loss.item(), 'Time:', time.time()-start_time)
                start_time = time.time()
            
            #if (i == 0) and (epoch+1 % 2 == 0):
            torch.save(self.ae_network.state_dict(), model_output_path + 'ae_epoch_e3'+str(epoch+1)+'.pth')
            torch.save(self.clf_network.state_dict(),  model_output_path + 'clf_epoch_e3'+str(epoch+1)+'.pth')
            print('Model saved')


    def test(self, model_load_path):
        dataloader = DataLoader(self.testdataset, self.batch_size, 
                                shuffle=True, num_workers=0)
        self.ae_network = Autoencoder().cuda()
        self.clf_network = clf_network().cuda()
        self.ae_network.load_state_dict(torch.load(model_load_path+'ae_epoch_e310.pth'))
        self.clf_network.load_state_dict(torch.load(model_load_path+'clf_epoch_e310.pth'))
        criterion = nn.CrossEntropyLoss().cuda()
        acc_list, f1_list, pre_list, recall_list, loss_list = [], [], [], [], []
        start_time = time.time()
        for i, data_label in enumerate(dataloader):
            data = Variable(data_label[0]).cuda().float()
            label = Variable(data_label[1]).view(-1).cuda().long()
            squeeze_data, _ = self.ae_network(data)          
            clf_pred = self.clf_network(squeeze_data)
            prob_out = clf_pred.detach().cpu().numpy()
            prob_idx = np.argmax(prob_out, 1)
            clf_loss = criterion(clf_pred, label)
            
            accuracy = metrics.accuracy_score(data_label[1], prob_idx)
            acc_list.append(accuracy)
            f1_score = metrics.f1_score(data_label[1], prob_idx)
            f1_list.append(f1_score)
            precision = metrics.precision_score(data_label[1], prob_idx)
            pre_list.append(precision)
            recall = metrics.recall_score(data_label[1], prob_idx)
            recall_list.append(recall)
            loss_list.append(clf_loss.item())

            #print('Iter', i, 'Time:', time.time()-start_time)
            print('Iter', i, 'Loss:', clf_loss.item(), 'Time:', time.time()-start_time)
            #print(accuracy, f1_score, precision, recall)
            start_time = time.time()

        print('accuracy:', np.mean(acc_list))
        print('f1_score:', np.mean(f1_list))
        print('precision:', np.mean(pre_list))
        print('recall', np.mean(recall_list))
        print('loss', np.mean(loss_list))



def main():
    LOCAL_PATH = "/mnt/nas/"
    MODEL_DIR = os.path.join(LOCAL_PATH, "PYUSCT_model/")
    DATA_DIR = os.path.join(LOCAL_PATH, "PYUSCT_train/")
    dataset_dir = os.path.join(DATA_DIR, "dataset036/")
    model_output_path = os.path.join(MODEL_DIR, "clf/deep/")
    #print(LOCAL_PATH,MODEL_DIR,model_output_path)
    scaler_path = os.path.join(MODEL_DIR, "Scaler/Log_MinMax_RFScaler_ds028.pickle")
    AE_weight_path = os.path.join(MODEL_DIR, "AE/rf_conv_AE_Log_MinMax_ds036.pth")
    
    model = clf(dataset_dir, scaler_path)
    #model.train(AE_weight_path, model_output_path)
    model.test(model_output_path)


if __name__ == '__main__':
    main()


