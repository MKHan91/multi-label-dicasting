import os
import random
import os.path as osp

import cv2
import numpy as np
import pandas as pd
import torchvision.transforms as transforms
import torch
from glob import glob
from torch.utils.data import Dataset
from torchvision.transforms import functional as F
from PIL import Image

import utils


class DatasetPreprocess:
    def __init__(self, data_cfg, test_cfg=None, mode='train'):
        self.image_paths = []
        self.data_cfg = data_cfg
        self.test_cfg = test_cfg
        
        if mode == 'train':
            self.image_paths = glob(osp.join(data_cfg.data_dir, mode, "Normal", "*.jpg"))

        elif mode == 'test':
            self.labels = []
            
            for class_name in test_cfg.target_classes:
                folder_dir = osp.join(data_cfg.data_dir, mode, class_name)
                for image_path in glob(osp.join(folder_dir, "*.jpg")):
                    self.image_paths.append(image_path)
                    self.labels.append(test_cfg.test_class_dict[class_name])
            
    def read_label_csv(self, csv_path: str):
        label_csv = pd.read_csv(csv_path)
        label_csv = label_csv[label_csv['label'].isin(['Normal', 'P', 'S'])]

        labels = label_csv[self.data_cfg.label_list].values.astype(np.float32)
        
        return labels


class diecastingDataset(Dataset):
    def __init__(self, data_cfg, test_cfg=None, mode='train'):
        preproceessor  = DatasetPreprocess(data_cfg, test_cfg, mode)
        
        self.image_paths = preproceessor.image_paths
        self.mode        = mode
        if mode == 'test':
            self.labels  = preproceessor.labels
        
        
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
        
        circles = circles.astype(np.uint8)
        cx, cy, r = circles[0][0]
        cx, cy, r = cx.item(), cy.item(), r.item()
        mask = np.zeros(image_arr.shape[:2], dtype=np.uint8)
        cv2.circle(mask, (cx, cy), r, 255, -1)

        image_arr = cv2.bitwise_and(image_arr, image_arr, mask=mask)
        x1, y1 = max(cx - r, 0)-5, max(cy - r, 0)-5
        x2 = min(cx + r, image_arr.shape[1] - 1) +5
        y2 = min(cy + r, image_arr.shape[0] - 1) +5
        
        bbox = [x1, y1, x2, y2]
        circle_center = [cx, cy]
        
        return image_arr, bbox, circle_center


    def apply_center_zoom(self, image: np.ndarray, bbox: list, circle_center: list):
        width, height = image.shape[:2]
        
        zoom_factor1 = abs(bbox[0] - bbox[2]) / width
        zoom_factor2 = abs(bbox[1] - bbox[3]) / height
        min_zoom = min(zoom_factor1, zoom_factor2)
        
        # zoom_factor = min_zoom
        zoom_factor = random.uniform(min_zoom, 1.0)
        
        new_h, new_w = int(height * zoom_factor), int(width * zoom_factor)
        cx, cy = circle_center
        p1_y = max(0, cy - new_h // 2)
        p2_y = min(height, cy + new_h // 2)
        p1_x = max(0, cx - new_w // 2)
        p2_x = min(width, cx + new_w // 2)

        cropped_image = image[p1_y:p2_y, p1_x:p2_x]
        image = cv2.resize(cropped_image, (width, height), interpolation=cv2.INTER_LINEAR)

        image = Image.fromarray(image)
        image = image.convert('RGB')
        
        return image
    
    
    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        image_name = osp.split(image_path)[-1][:-4]
        image = Image.open(image_path).convert('L')
        
        image, bbox, circle_center = self.get_circle_roi(image)
        if self.mode=='train':
            image = self.apply_center_zoom(image, bbox, circle_center)
        
            data_transforms = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.RandomHorizontalFlip(),             
                transforms.RandomVerticalFlip(),               
                transforms.RandomRotation(45),                 
                transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
                # transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2),  
                transforms.ToTensor(),
                # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                ])
            
            image = data_transforms(image)
        
            return image, image_name
        
        elif self.mode=='test':
            image = Image.fromarray(image)
            image = image.convert('RGB')
            
            data_transforms = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                ])
            
            image = data_transforms(image)
            label   = self.labels[idx]
            
            return image, label, image_name
        

    def get_labels(self):
        return self.labels