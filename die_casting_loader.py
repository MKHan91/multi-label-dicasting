import os
import random
import os.path as osp

import cv2
import numpy as np
import pandas as pd
import torchvision.transforms as transforms

from glob import glob
from torch.utils.data import Dataset
from torchvision.transforms import functional as F
from PIL import Image



class castingDatasetPreprocess:
    def __init__(self, data_dir, mode='train'):
        if mode == 'train':
            self.image_paths = []
            for folder_name in sorted(os.listdir(data_dir)):
                if not osp.isdir(osp.join(data_dir, folder_name)): continue
                
                image_paths = glob(osp.join(data_dir, folder_name, "*.jpg"))
                self.image_paths += image_paths
                
            csv_path                    = osp.join(data_dir, 'labels.csv')
            self.labels, self.densities = self.read_label_csv(csv_path)
        
        elif mode == 'test':
            self.image_paths  = glob(osp.join(data_dir, mode, '*.png'))
        
        
    def read_label_csv(self, csv_path):
        label_csv = pd.read_csv(csv_path)
        
        label_columns = ['label_P', 'label_S', 'label_IMC']
        train_labels = label_csv[label_columns].values.astype(np.float32)
        train_densities = label_csv['density'].values.astype(np.float32)
        
        return train_labels, train_densities


class castingDataset(Dataset):
    def __init__(self, data_dir, mode='train'):
        data_preprocess  = castingDatasetPreprocess(data_dir)
        self.image_paths = data_preprocess.image_paths
        self.labels      = data_preprocess.labels
        self.densities   = data_preprocess.densities
        self.mode        = mode
        
        
    def __len__(self):
        return len(self.image_paths)


    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        
        image = Image.open(image_path).convert('RGB')

        # 데이터 증강
        if self.mode=='train':
            data_transforms = transforms.Compose([
                transforms.RandomHorizontalFlip(),             
                transforms.RandomVerticalFlip(),               
                transforms.RandomRotation(45),                 
                transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
                transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2),  
                transforms.RandomResizedCrop(512),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
            
            image = data_transforms(image)
        
        
        elif self.mode=='test':
            data_transforms = transforms.Compose([
                transforms.Resize((512, 512)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
            
            image = data_transforms(image)

        label   = self.labels[idx]
        densities = self.densities[idx]
        
        return image, label, densities


    def get_labels(self):
        return self.labels