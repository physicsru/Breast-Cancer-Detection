import os, sys, glob, math
sys.path.append("../../preprocess/pyusct/")
from rfdata import RFdata
from AE import Autoencoder, RFCompressedDataset
#from network import clf_network

import numpy as np
from sklearn import metrics

import torch, torchvision
from torch import nn
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from sklearn.model_selection import train_test_split
import pickle, time
sys.path.append("../model/")
from model import clf_network



class clf():
    def __init__(self, dataset_dir):
        self.lr = 1e-2
        self.epochs = 100000
        self.batch_size = 32
        self.input_list = sorted(glob.glob(os.path.join(dataset_dir, "input/*.npy")))
        self.output_list = sorted(glob.glob(os.path.join(dataset_dir, "output/*.npy")))
        # X_train, X_test, y_train, y_test = train_test_split(self.input_list, self.output_list, test_size=0.2, random_state=42)
        # self.traindataset = RFCompressedDataset(X_train, y_train)
        # self.testdataset = RFCompressedDataset(X_test, y_test)
        self.traindataset = RFCompressedDataset(self.input_list[:-1], self.output_list[:-1])
        self.testdataset = RFCompressedDataset(self.input_list[-1:], self.output_list[-1:])

    def train(self, model_output_path):
        
        dataloader = DataLoader(self.traindataset, self.batch_size, 
                                shuffle=True, num_workers=0)
        
        self.clf_network = clf_network().cuda()
        
        all_params = list(self.clf_network.parameters())
        
        optimizer = torch.optim.Adam(all_params, lr=self.lr, weight_decay=1e-4)
        criterion = nn.CrossEntropyLoss().cuda()
        print('start training')
        
        for epoch in range(self.epochs):
            start_time = time.time()
            for i, data_label in enumerate(dataloader):
                data = Variable(data_label[0]).cuda().float()
                label = Variable(data_label[1]).view(-1).cuda().long()
                
                clf_pred = self.clf_network(data)
                clf_loss = criterion(clf_pred, label)

                optimizer.zero_grad()
                clf_loss.backward()
                optimizer.step()

                if i % 100 == 0:
                    print('Epoch:', epoch, 'Iter', i, 'Loss:', clf_loss.item(), 'Time:', time.time()-start_time)
                start_time = time.time()
            
            if epoch % 10000 == 0:
                torch.save(self.clf_network.state_dict(),  model_output_path + 'clf_pca_epoch_'+str(epoch)+'.pth')
                print('Model saved')


    def test(self, model_load_path):
        dataloader = DataLoader(self.testdataset, self.batch_size, 
                                shuffle=True, num_workers=0)
        self.clf_network = clf_network().cuda()
        self.clf_network.load_state_dict(torch.load(model_load_path+'clf_pca_epoch_30000.pth'))
        criterion = nn.CrossEntropyLoss().cuda()
        acc_list, f1_list, pre_list, recall_list, loss_list = [], [], [], [], []
        start_time = time.time()
        for i, data_label in enumerate(dataloader):
            data = Variable(data_label[0]).cuda().float()
            label = Variable(data_label[1]).view(-1).cuda().long()
            clf_pred = self.clf_network(data)
            prob_out = clf_pred.detach().cpu().numpy()
            prob_idx = np.argmax(prob_out, 1)
            clf_loss = criterion(clf_pred, label)
            loss_list.append(clf_loss.item())
            accuracy = metrics.accuracy_score(data_label[1], prob_idx)
            acc_list.append(accuracy)
            f1_score = metrics.f1_score(data_label[1], prob_idx)
            f1_list.append(f1_score)
            precision = metrics.precision_score(data_label[1], prob_idx)
            pre_list.append(precision)
            recall = metrics.recall_score(data_label[1], prob_idx)
            recall_list.append(recall)

            print('Iter', i, 'Time:', time.time()-start_time)
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
    dataset_dir = os.path.join(DATA_DIR, "dataset037/")
    model_output_path = os.path.join(MODEL_DIR, "clf/deep/compression_fix")
    #print(LOCAL_PATH,MODEL_DIR,model_output_path)
    model = clf(dataset_dir)
    #model.train(model_output_path)
    model.test(model_output_path)


if __name__ == '__main__':
    main()