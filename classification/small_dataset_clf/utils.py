import cv2
import os
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from skimage import exposure

'''
dataloader with ramdon sample for data agmentation
'''
def next_batch(data, label, batch_size):
    idx = np.arange(0 , len(data))
    np.random.shuffle(idx)
    idx = idx[:batch_size]
    batch_data = [data[i] for i in idx]
    batch_label = [label[i] for i in idx]
    
    return batch_data, batch_label






