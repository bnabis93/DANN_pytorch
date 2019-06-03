import torch.utils.data as data
from PIL import Image
import os
import numpy as np

'''
DANN dataloader

for mnist, mnist-m , etc...

Reference : https://github.com/fungtion/DANN/blob/master/dataset/data_loader.py
'''

class GetLoader(data.Dataset):
    def __init__(self, data_root, data_list, transform = None):
        '''
        self params
        
        self.root
            path of data
            dataType = str
        self.transform
            also path of data (specific data path. like train, test)
            dataType = str
        self.n_data
            number of data (data list)
            dataType = int
        self.img_paths
            
        self.img_labels
        
        '''
        self.root = data_root
        self.transform = transform
        
        # file read
        f = open(data_list, 'r')
        data_list = f.readlines()
        f.close()
        
        self.n_data = len(data_list)
        #self.n_data = data_list
        #print(len(self.n_data))
        
        self.img_paths = []
        self.img_labels = []
        
        #print('shape of data_list', np.shape(data_list))
        
        for data in data_list:
            self.img_paths.append(data[:-3])
            self.img_labels.append(data[-2])
            
        #print('shape of img_path', np.shape(self.img_paths))
        #print('shape of img_label', np.shape(self.img_labels))
            
    def __getitem__(self, item):
        img_paths, labels = self.img_paths[item], self.img_labels[item]
        imgs = Image.open(os.path.join(self.root, img_paths)).convert('RGB')
            
        if self.transform is not None:
            imgs = self.transform(imgs)
            labels = int(labels)
            
        return imgs,  labels
        
    def __len__(self):
        return self.n_data
            
            