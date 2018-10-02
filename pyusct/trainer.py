import os, sys, glob, time, math, json
sys.path.append("../pyusct/")

import numpy as np
from sklearn import metrics
from sklearn.model_selection import train_test_split

import torch
from torch import nn
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader

from model import Conv_2d, Conv_3d, Conv_3d_VGG, Conv_3d_SA, Conv_3d_VGG_SA, Conv_3d_transfer, Conv_3d_VGG_transfer, Conv_3d_VGG_parallel
from pytorch_dataset import RFFullDataset, RFFullDataset3d

import matplotlib.pyplot as plt

class Trainer_prototype():
    
    def __init__(self, dataset_dir, model_output_path, lr, epochs, batch_size, l2_alpha, type, random_state):
        
        timestr = time.strftime("%Y%m%d-%H%M%S")
        self.output_path = os.path.join(model_output_path, timestr + "/")
        
        params = {
            "dataset": dataset_dir,
            "model_type": type,
            "lr": lr,
            "epoch": epochs,
            "batchsize":batch_size,
            "l2_alpha": l2_alpha,
            "random_state": random_state,
        }
        params = json.dumps(params)
        
        if not os.path.exists(self.output_path):
            os.makedirs(self.output_path)
        # save params
        with open(os.path.join(self.output_path, 'params.json'), 'w') as outfile:
            json.dump(params, outfile)
        # initialization
        self.lr, self.epochs, self.batch_size, self.l2_alpha, self.type = lr, epochs, batch_size, l2_alpha, type
        self.input_list = sorted(glob.glob(os.path.join(dataset_dir, "input/*.npy")))
        if "SA" in self.type:
            self.output_list = sorted(glob.glob(os.path.join(dataset_dir, "SA/*.npy")))
        else:
            self.output_list = sorted(glob.glob(os.path.join(dataset_dir, "output/*.npy")))
            
        X_train, X_test, y_train, y_test = train_test_split(self.input_list, self.output_list, test_size=0.2, random_state=42)
        X_train, X_valid, y_train, y_valid = train_test_split(X_train, y_train, test_size=0.125, random_state=42)
        self.train_dataset = RFFullDataset3d(X_train, y_train, self.model.scaler)
        self.valid_dataset = RFFullDataset3d(X_valid, y_valid, self.model.scaler)
        self.test_dataset = RFFullDataset3d(X_test, y_test, self.model.scaler)
        
        self.train_loss = []
        self.valid_loss = []
        return
    
    def train(self):
        dataloader_train = DataLoader(self.train_dataset, self.batch_size, 
                                shuffle=True)
        
        dataloader_valid = DataLoader(self.valid_dataset, self.batch_size, 
                                shuffle=True)
        all_params = self.model.get_params()
        optimizer = torch.optim.Adam(all_params, lr=self.lr, weight_decay=self.l2_alpha)
        criterion = None
        if "SA" in self.type:
            criterion = nn.MSELoss().cuda()
        else:
            criterion = nn.CrossEntropyLoss().cuda()
            
        print('start training')
        
        for epoch in range(self.epochs):
            start_time = time.time()
            train_loss = []
            valid_loss = []
            
            # train
            for i, (data, label) in enumerate(dataloader_train):
                data = Variable(data).cuda().float()
                if "SA" in self.type:
                    label = Variable(label).view(-1, 1).cuda().float()
                else:
                    label = Variable(label).view(-1).cuda().long()
                
                pred = self.model.pred(data)
                clf_loss = criterion(pred, label)
                train_loss.append(clf_loss.item())
                optimizer.zero_grad()
                clf_loss.backward()
                optimizer.step()

                if (i+1) % 10 == 0:
                    print('Epoch:', epoch, 'Iter', i, 'Loss:', clf_loss.item(), 'Time:', time.time()-start_time)
                start_time = time.time()
                
            # valid
            for i, (data, label) in enumerate(dataloader_valid):
                data = Variable(data).cuda().float()
                label = Variable(label).cuda().float()
                if "SA" in self.type:
                    label = Variable(label).view(-1, 1).cuda().float()
                else:
                    label = Variable(label).view(-1).cuda().long()
                    
                pred = self.model.pred(data)
                
                clf_loss = criterion(pred, label)
                valid_loss.append(clf_loss.item())
            
            self.train_loss.append(np.mean(train_loss))
            self.valid_loss.append(np.mean(valid_loss))
            
            print("Epoch: {}, train_loss: {}, valid_loss: {}".format(epoch, self.train_loss[-1], self.valid_loss[-1]))
                
            if (epoch+1) % 2 == 0:
                name = 'raw_data_epoch_'+str(epoch)+'.pth'
                self.model.save_model(self.output_path, self.type, name)
                print('Model saved')
        return


    def test(self, model_params=None):
        dataloader = DataLoader(self.test_dataset, self.batch_size, 
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

        print('accuracy:', round(np.mean(acc_list), 4))
        print('f1_score:', round(np.mean(f1_list), 4))
        print('precision:', round(np.mean(pre_list), 4))
        print('recall', round(np.mean(recall_list), 4))
        print('loss', round(np.mean(loss_list), 4))
        return
        
    def plot_learn_curve(self):
        plt.plot(self.train_loss)
        plt.plot(self.valid_loss)
        plt.show()
        return 
    
    
class Trainer_3d_transfer(Trainer_prototype):
    
    def __init__(self, dataset_dir, scaler_path, model_output_path, lr=1e-3, epochs=100, batch_size=32, l2_alpha=1e-3, type="3d_transfer", random_state=42, params=None):
        
        if params:
            self.model = Conv_3d_transfer(scaler_path, params)
        else:
            self.model = Conv_3d_transfer(scaler_path)
            
        Trainer_prototype.__init__(self, dataset_dir, model_output_path, lr, epochs, batch_size, l2_alpha, type, random_state)
            
        return
    
    
class Trainer_3d_VGG_transfer(Trainer_prototype):
    
    def __init__(self, dataset_dir, scaler_path, model_output_path, lr=1e-3, epochs=100, batch_size=8, l2_alpha=1e-3, type="3d_VGG_transfer", random_state=42, params=None):
        
        if params:
            self.model = Conv_3d_VGG_transfer(scaler_path, params)
        else:
            self.model = Conv_3d_VGG_transfer(scaler_path)
        
        Trainer_prototype.__init__(self, dataset_dir, model_output_path, lr, epochs, batch_size, l2_alpha, type, random_state)
        
        return 
    
    
class Trainer_3d_SA(Trainer_prototype):
    
    def __init__(self, dataset_dir, scaler_path, model_output_path, lr=1e-3, epochs=100, batch_size=16, l2_alpha=1e-3, type="3d_SA", random_state=42):
        
        self.model = Conv_3d_SA(scaler_path)
        
        Trainer_prototype.__init__(self, dataset_dir, model_output_path, lr, epochs, batch_size, l2_alpha, type, random_state)
        
        return     

    
class Trainer_3d_VGG_SA(Trainer_prototype):
    
    def __init__(self, dataset_dir, scaler_path, model_output_path, lr=1e-3, epochs=100, batch_size=8, l2_alpha=1e-3, type="3d_VGG_SA", random_state=42):
        
        self.model = Conv_3d_VGG_SA(scaler_path)
        
        Trainer_prototype.__init__(self, dataset_dir, model_output_path, lr, epochs, batch_size, l2_alpha, type, random_state)
        
        return 

    
class Trainer_3d_VGG(Trainer_prototype):
    
    def __init__(self, dataset_dir, scaler_path, model_output_path, lr=1e-3, epochs=100, batch_size=4, l2_alpha=1e-3, type="3d_VGG", random_state=42):
        
        self.model = Conv_3d_VGG(scaler_path)
        
        Trainer_prototype.__init__(self, dataset_dir, model_output_path, lr, epochs, batch_size, l2_alpha, type, random_state)
        
        return 
    
    
class Trainer_3d(Trainer_prototype):
    
    def __init__(self, dataset_dir, scaler_path, model_output_path, lr=1e-3, epochs=100, batch_size=8, l2_alpha=1e-3, type="3d", random_state=42):
        
        self.model = Conv_3d(scaler_path)
        
        Trainer_prototype.__init__(self, dataset_dir, model_output_path, lr, epochs, batch_size, l2_alpha, type, random_state)
        
        return 
    
class Trainer_3d_VGG_parallel(Trainer_prototype):
    
    def __init__(self, dataset_dir, scaler_path, model_output_path, lr=1e-3, epochs=100, batch_size=16, l2_alpha=1e-3, type="3d_VGG", random_state=42):
        
        self.model = Conv_3d_VGG_parallel(scaler_path)
        
        Trainer_prototype.__init__(self, dataset_dir, model_output_path, lr, epochs, batch_size, l2_alpha, type, random_state)
        
        return 
  
    

    