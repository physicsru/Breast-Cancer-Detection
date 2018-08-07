import cv2
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from skimage import exposure

'''
dataloader with ramdon sample for data agmentation
'''
def next_batch(data, point_label, img_size, batch_size):
    batch_data, batch_img_label, batch_point_label = [], [], []
    total_h, total_w = 512, 512
    total_num = len(data)
    for i in range(batch_size):
        idx = np.random.randint(0, total_num)
        offset_h = np.random.randint(0, total_h - img_size)
        offset_w = np.random.randint(0, total_w - img_size)
        
        img_label_crop = point_label[idx][offset_h: offset_h + img_size, 
                                        offset_w: offset_w + img_size]*255
        batch_img_label.append(img_label_crop)
        
        point_label_crop = point_label[idx][offset_h: offset_h + img_size, 
                                        offset_w: offset_w + img_size]
        batch_point_label.append(np.reshape(point_label_crop, (img_size*img_size, 1)))
        
        data_crop = data[idx][offset_h: offset_h + img_size, 
                                        offset_w: offset_w + img_size]
        batch_data.append(np.reshape(data_crop, (-1, 800)))
        
    batch_point_label = np.asarray(batch_point_label)
    onehot_encoder = OneHotEncoder(sparse=False)
    batch_point_label = onehot_encoder.fit_transform(batch_point_label.reshape(-1, 1))
    
    batch_data = np.asarray(batch_data)
    batch_data = np.reshape(batch_data, (-1, 800))
    
    return batch_data, batch_img_label, batch_point_label


'''
pay attention to the data folder path and data format
'''

def load_data(low, high):
    train_data = []
    for i in range(low, high):
        dir1 = 'dataset012/trial_' + str(i).zfill(3) + '/input'
        train_data_sep = []
        for j in range(0, 4):
            dir2 = dir1 + '/part' + str(j).zfill(3) + '_size65536.npy'
            train_data_sep.append(np.load(dir2))
        train_data_sep = np.concatenate(train_data_sep, axis=0)
        train_data_sep = np.reshape(train_data_sep, (512, 512, -1))
        train_data.append(train_data_sep)
    train_data = np.asarray(train_data)
    return train_data
    

def load_label(low, high):
    train_label = []
    for i in range(low, high):
        dir1 = 'dataset012/trial_' + str(i).zfill(3) + '/output/label_size_512m512.npy'
        train_label_sep = np.load(dir1)
        train_label_sep = np.concatenate(train_label_sep, axis=0)
        train_label_sep = np.reshape(train_label_sep, (512, 512, -1))
        train_label.append(train_label_sep)
    train_label = np.asarray(train_label)
    return train_label


'''
Memory is not big enough for the dataset, so we randomly load part of it,
and change the parts every 1000 iterations.
'''

def load_data_label():
    train_data, train_label = [], []
    idx = np.arange(2, 14)
    np.random.shuffle(idx)
    idx = idx[: 6]
    for i in idx:
        dir_data = 'dataset012/trial_' + str(i).zfill(3) + '/input'
        train_data_sep = []
        for j in range(0, 4):
            data_file_name = dir_data + '/part' + str(j).zfill(3) + '_size65536.npy'
            train_data_sep.append(np.load(data_file_name))
        train_data_sep = np.concatenate(train_data_sep, axis=0)
        train_data_sep = np.reshape(train_data_sep, (512, 512, -1))
        train_data.append(train_data_sep)
        
        dir_label = 'dataset012/trial_' + str(i).zfill(3) + '/output/label_size_512m512.npy'
        train_label_sep = np.load(dir_label)
        train_label_sep = np.concatenate(train_label_sep, axis=0)
        train_label_sep = np.reshape(train_label_sep, (512, 512, -1))
        train_label.append(train_label_sep)
        
    train_data, train_label = np.asarray(train_data), np.asarray(train_label)
    return train_data, train_label
        
        
    
 
def load_test_data():
    train_data = []
    for i in range(14, 17):
        dir1 = 'dataset012/trial_' + str(i).zfill(3) + '/input'
        train_data_sep = []
        for j in range(0, 4):
            dir2 = dir1 + '/part' + str(j).zfill(3) + '_size65536.npy'
            train_data_sep.append(np.load(dir2))
        train_data_sep = np.concatenate(train_data_sep, axis=0)
        train_data_sep = np.reshape(train_data_sep, (512, 512, -1))
        train_data.append(train_data_sep)
    train_data = np.asarray(train_data)
    return train_data



''' 
for i in range(2, 17):
    dir1 = 'dataset012/trial_' + str(i).zfill(3) + '/output/label_size_512m512.npy'
    train_label_sep = np.load(dir1)
    train_label_sep = np.concatenate(train_label_sep, axis=0)
    train_label_sep = np.reshape(train_label_sep, (512, 512, -1))
    img = exposure.rescale_intensity(train_label_sep, out_range=(0, 255))
    #print(np.shape(img))
    cv2.imwrite(str(i).zfill(3)+'_output.bmp', img)
'''