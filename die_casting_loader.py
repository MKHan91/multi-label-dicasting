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


class DatasetPreprocess:
    def __init__(self, data_cfg, mode='train'):
        self.image_paths = []
        self.data_cfg = data_cfg
        
        if mode == 'train':
            self.image_paths = glob(osp.join(data_cfg.data_dir, mode, "Normal", "*.jpg"))
            
        elif mode == 'test':
            folder_dir = [
                osp.join(data_cfg.data_dir, mode, "Normal"),
                osp.join(data_cfg.data_dir, mode, "P"),
                osp.join(data_cfg.data_dir, mode, "S")
            ]
            for dir in folder_dir:
                self.image_paths.extend(glob(osp.join(dir, "*.jpg")))
                
        csv_path = osp.join(data_cfg.data_dir, mode, f'{data_cfg.label_csv_name}.csv')
        self.labels, self.densities = self.read_label_csv(csv_path)
    
    
    def read_label_csv(self, csv_path: str):
        label_csv = pd.read_csv(csv_path)
        label_csv = label_csv[label_csv['label'].isin(['Normal', 'P', 'S'])]

        train_labels = label_csv[self.data_cfg.label_list].values.astype(np.float32)
        train_densities = label_csv['density'].values.astype(np.float32)
        
        return train_labels, train_densities


class diecastingDataset(Dataset):
    def __init__(self, data_cfg, mode='train'):
        preproceessor  = DatasetPreprocess(data_cfg, mode)
        self.image_paths = preproceessor.image_paths
        self.labels      = preproceessor.labels
        self.densities   = preproceessor.densities
        
        self.mode        = mode
        
    def __len__(self):
        return len(self.image_paths)


    def get_circle_roi(self, image: Image.Image):
        image_arr = np.array(image)
        circles = cv2.HoughCircles(image_arr, cv2.HOUGH_GRADIENT,
                                    dp=1.2,
                                    minDist=50,
                                    param1=100,   # Canny high threshold
                                    param2=30,    # 원 판별 threshold (낮을수록 민감)
                                    minRadius=30,
                                    maxRadius=150)
        
        circles = np.uint16(np.around(circles))
        cx, cy, r = circles[0][0]
        mask = np.zeros(image_arr.shape[:2], dtype=np.uint8)
        cv2.circle(mask, (cx, cy), r, 255, -1)

        image_arr = cv2.bitwise_and(image_arr, image_arr, mask=mask)
        image = Image.fromarray(image_arr)
        image = image.convert('RGB')
        
        return image
    
    
    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        image_name = osp.split(image_path)[-1][:-4]
        image = Image.open(image_path).convert('L')
        # image = image.crop((0, 40, image.size[0], 235))
        image = self.get_circle_roi(image)
        
        # circle mask 적용
        # 데이터 증강
        if self.mode=='train':
            # center zoom & center crop
            data_transforms = transforms.Compose([
                transforms.CenterCrop(224),  
                transforms.RandomHorizontalFlip(),             
                transforms.RandomVerticalFlip(),               
                transforms.RandomRotation(45),                 
                transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
                transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2),  
                # transforms.RandomResizedCrop(224),
                transforms.ToTensor(),
                # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                ])
            
            image = data_transforms(image)
        
            return image, image_name
        
        elif self.mode=='test':
            data_transforms = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                ])
            
            image = data_transforms(image)

            label   = self.labels[idx]
            densities = self.densities[idx]
            
            return image, label, image_name
        


    def get_labels(self):
        return self.labels