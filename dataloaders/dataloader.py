import os
import glob
import numpy as np
import torch
import h5py
import yaml
import cv2
import random
import time
from tqdm import tqdm

from scipy.spatial.distance import cdist
from torch.utils.data import Dataset

class LineDescDataset(Dataset):
    def __init__(self, data_path):
        self.listing = []
        for path in data_path:
            h5py_glob = '*.h5'
            search = os.path.join(path, h5py_glob)
            self.listing.extend(glob.glob(search))
        # self.listing = self.listing[:]

    def __len__(self):
        return len(self.listing)

    def __getitem__(self, index):
        path = self.listing[index]
        
        batch_data = {}
        with h5py.File(path, "r") as f:
            for name in f:
                try:
                    batch_data[name] = f[name][()]
                except:
                    print('here')
                
        assert len(batch_data) != 0, path

        return batch_data

# class DatasetManager():
#     def __init__(self, data_path, num_samples_per_block = None):
#         # list up all dataset
#         listing = []
#         for path in data_path:
#             h5py_glob = '*.h5'
#             search = os.path.join(path, h5py_glob)
#             listing.extend(glob.glob(search))
        
#         random.shuffle(listing)

#         if num_samples_per_block == None:
#             self.dataset = listing
#         else:
#             sub_listing = [listing[x:x+num_samples_per_block] for x in range(0, len(listing), num_samples_per_block)]
#             self.dataset = sub_listing
    
#     def shuffle(self):
#         self.dataset = sorted(self.dataset ,key = lambda i:random.random())

#         # # random.shuffle(self.dataset)
#         # # self.dataset = random.sample(self.dataset, len(self.dataset)) 
#         # for i in range(len(self.dataset)-1, 0, -1): 
      
#         #     # Pick a random index from 0 to i  
#         #     j = random.randint(0, i + 1)  
            
#         #     # Swap arr[i] with the element at random index  
#         #     self.dataset[i], self.dataset[j] = self.dataset[j], self.dataset[i]  

# class LineDescDataset2(Dataset):
#     def __init__(self, data_list):
#         self.listing = data_list

#     def __len__(self):
#         return len(self.listing)

#     def __getitem__(self, index):
#         path = self.listing[index]
        
#         batch_data = {}
#         with h5py.File(path, "r") as f:
#             for name in f:
#                 # if name == 'angle_klines0' or name == 'angle_klines1':
#                 #     continue
#                 # batch_data[name] = f[name][:][0]
#                 if name == 'is_valid_data' or name == 'scene_id':
#                     continue
#                 batch_data[name] = f[name][:]
#                 # print(name, batch_data[name].shape)
                
#         assert len(batch_data) != 0, path

#         return batch_data

#     def shuffle_dataset(self):
#         random.shuffle(self.listing)