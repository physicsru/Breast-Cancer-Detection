import numpy as np
import util
import cv2
from sklearn.decomposition import PCA
import pickle
import os, sys, glob, json, time
sys.path.append("./pyusct/")
from rfdata import RFdata

LOCAL_PATH = "/media/yuhui/dea78678-112b-4f0f-acbf-4e9d1be35e35/nas/"
MOUNT_PATH = "/run/user/1000/gvfs/smb-share:server=azlab-fs01,share=東研究室/個人work/富井/"
MODEL_DIR = "/run/user/1000/gvfs/smb-share:server=azlab-fs01,share=東研究室/個人work/富井/PYUSCT_model/PCA/"

class PCATrainer(object):
    
    def __init__(self, input_shape=(16, 256, 200), output_shape=(800)):
        self.input_shape = input_shape
        self.output_shape = output_shape
        return 
    
    def train(self, data):
        self.pca = PCA(n_components=self.output_shape)
        self.pca.fit(data)
        evr = self.pca.explained_variance_ratio_
        cum_evr = np.cumsum(evr)
        np.where(cum_evr>0.995)[0][0]
        return self.pca
    
    def save_model(self, model_name):
        if self.pca==None: return
        with open(os.path.join(MODEL_DIR, model_name + '.pickle'), 'wb') as handle:
            pickle.dump(self.pca, handle, protocol=pickle.HIGHEST_PROTOCOL)
        return 
    
    def debug(self):
        pass
        
    
    
    