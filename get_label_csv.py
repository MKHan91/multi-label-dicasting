import os
import os.path as osp
import pandas as pd
import numpy as np

from configuration import DataConfig as data_cfg
from configuration import BaseConfig as cfg


label_map = {
    "Normal": [0, 0, 0],
    "P"     : [1, 0, 0],
    "S"     : [0, 1, 0],
    "PS"    : [1, 1, 0],
    "IMC"   : [0, 0, 1],
    "S_IMC" : [0, 1, 1],
    "P_IMC" : [1, 0, 1],
    "PS_IMC": [1, 1, 1],
}

density_map = {
    "Normal": 2.8,
    "P":   0.5,
    "S":   0.5,
    "PS":  0.5,
    "IMC": 3.3
    
}
# 데이터 경로
base_dir = "/home/dev/multi-label-dicasting/dataset"
data_dir = osp.join(base_dir, cfg.mode)

classes = sorted(os.listdir(data_dir))

img_labels = []
for class_idx, class_name in enumerate(classes):
    class_dir = osp.join(data_dir, class_name)
    if not osp.isdir(class_dir): continue
    
    image_names = [image_name[:-4] for image_name in os.listdir(class_dir) if class_name in label_map]
    
    for image_name in image_names:
        label = label_map[class_name]
        density = density_map[class_name]
        
        img_labels.append({
            'fileName': image_name,
            'label': class_name,
            'P': label[0],
            'S': label[1],
            'IMC': label[2],
            'density': density
        })

df = pd.DataFrame(img_labels)
df.to_csv(osp.join(base_dir, cfg.mode, f"{data_cfg.label_csv_name}.csv"), index=False)
print('저장 완료')