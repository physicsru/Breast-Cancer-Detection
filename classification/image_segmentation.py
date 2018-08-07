# -*- coding: utf-8 -*-
"""
Created on Thu Feb 22 06:16:34 2018

@author: System_Error
"""

import cv2
import numpy as np
import os
from tqdm import tqdm
from skimage import exposure, filters
from sklearn.cluster import KMeans  

load_dir = 'breast_super_sonic'
save_dir = 'breast_img'
num_class = 3

if not os.path.exists(save_dir):
    os.makedirs(save_dir)
    
'''
images = []
for filename in os.listdir(load_dir):
    img = cv2.imread(os.path.join(load_dir, filename), 0)
    images.append(img)

    
for _ in range(5):
    mean = np.mean(images)
    images = images - mean
    images = images*(255/images.max())


empty_img = 255*np.ones((512, 512), np.uint8)
cv2.circle(empty_img, (255, 255), 200, 0, -1)

for i in tqdm(range(len(images))):
    images[i] = images[i] + empty_img
    for j in range(512):
        for k in range(512):
            if empty_img[j][k]==255:
                images[i][j][k] = 0
    cv2.imwrite(os.path.join(save_dir, str(i).zfill(4)+'.bmp'), images[i])
        
    '''
    

for i in tqdm(range(266)):
    img = cv2.imread(os.path.join(save_dir, str(i).zfill(4)+'.bmp'), 0)
    cv2.imwrite(str(i).zfill(4)+'00_origin1.bmp', img)
    
    img = exposure.rescale_intensity(img)
    '''
    clahe = cv2.createCLAHE()
    equal_img = clahe.apply(img)
    '''
    img = cv2.equalizeHist(img)
    cv2.imwrite(str(i).zfill(4)+'00_normalized.bmp', img)
    
    thresh = 1.2 * filters.threshold_otsu(img)
    ret1, thresh1 = cv2.threshold(img, thresh, 255, cv2.THRESH_TOZERO)
    thresh1 = exposure.rescale_intensity(thresh1)
    
    
    #ad_thresh = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 75, 2)
    
    #cv2.imwrite(str(i).zfill(4)+'00_thresh_result1.bmp', thresh1)
   
    
    thresh = 1.35*filters.threshold_otsu(thresh1)
    ret2,thresh2 = cv2.threshold(thresh1, thresh, 255, cv2.THRESH_TOZERO)
    thresh2 = exposure.rescale_intensity(thresh2)
    #cv2.imwrite(str(i).zfill(4)+'02_thresh_result2.bmp', thresh2)
    
    thresh = 1.5*filters.threshold_otsu(thresh2)
    ret3,thresh3 = cv2.threshold(thresh2, thresh, 255, cv2.THRESH_TOZERO)
    thresh3 = exposure.rescale_intensity(thresh3)
    #cv2.imwrite(str(i).zfill(4)+'03_thresh_result3.bmp', thresh3)
    

    row, col = np.shape(img)[0], np.shape(img)[1]
    
    label = KMeans(num_class).fit_predict(np.reshape(thresh3, (-1, 1)))
    max_label = np.argmax(np.bincount(label))
    min_label = np.argmin(np.bincount(label))
    label = label.reshape([row,col])  
    
    #label = exposure.rescale_intensity(label)
    
    pic_new = np.empty((row, col), int)
    
    for p in range(row):                        
        for q in range(col):  
            
            if label[p][q] == max_label or  label[p][q] == min_label:
                pic_new[p][q] = 0
            else:
                pic_new[p][q] = thresh3[p][q]
            '''
            if label[p][q] == 0:
                pic_new[p][q] = 0
            elif label[p][q] == 1:
                pic_new[p][q] = 127
            elif label[p][q] == 2:
                pic_new[p][q] = 255
            '''
            #pic_new[p][q] = label[p][q] * 255 / num_class
    
    #cv2.imwrite(str(i).zfill(4)+'05_enhance_contrast3.bmp', enhence3)
    #connect = measure.label(pic_new, connectivity=2)
    #cv2.imwrite(str(i).zfill(4)+'05_connect.bmp', connect)
    cv2.imwrite(str(i).zfill(4)+'05_kmeans.bmp', pic_new)
    
    


