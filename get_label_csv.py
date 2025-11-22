import os
import os.path as osp
import pandas as pd
from configuration import DataConfig as data_cfg
from configuration import BaseConfig as cfg


label_map = {
    "Normal": [0, 0, 0],
    "P"     : [1, 0, 0],
    "S"     : [0, 1, 0],
    "PS"    : [1, 1, 0],
    "IMC"   : [0, 0, 1],
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
data_dir = osp.join(base_dir, "train")

classes = sorted(os.listdir(data_dir))
print(f'분류 클래스: {classes}')

# 이미지별 라벨 저장 리스트
img_labels = []

# 각 폴더를 순회(반복)하면서 라벨 부여
for class_idx, class_name in enumerate(classes):
    class_dir = osp.join(data_dir, class_name)
    if not osp.isdir(class_dir): continue
    
    for image_name in os.listdir(class_dir):
        if class_name not in label_map: continue
        
        vector = label_map[class_name]
        density = density_map[class_name]
        img_labels.append(
            {"fileName": image_name,
             "label": class_name,
             "P": vector[2],
             "S": vector[1],
             "IMC": vector[0],
             "density": density
             }
            
        )

df = pd.DataFrame(img_labels)
df.to_csv(osp.join(base_dir, cfg.mode, f"{data_cfg.label_csv_name}.csv"), index=False)
print('저장 완료')