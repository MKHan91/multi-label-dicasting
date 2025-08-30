import os
import os.path as osp
import pandas as pd

# multi-hot encoding
# PS, P, S, N, IMC
# N: [0, 0, 0]
# P: [0, 0, 1]
# S: [0, 1, 0]
# PS: [0, 1, 1]
# IMC: [1, 0, 0]

label_map = {
    "Normal": [0, 0, 0],
    "P":   [0, 0, 1],
    "S":   [0, 1, 0],
    "PS":  [0, 1, 1],
    "IMC": [1, 0, 0],
}

density_map = {
    "Normal": 2.8,
    "P":   0.5,
    "S":   0.5,
    "PS":  0.5,
    "IMC": 3.3
    
}
# 데이터 경로
data_dir = "/home/dev/DATASET/die_casting"

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
             "label_P": vector[2],
             "label_S": vector[1],
             "label_IMC": vector[0],
             "density": density
             }
            
        )


df = pd.DataFrame(img_labels)
df.to_csv("/home/dev/DATASET/die_casting/labels.csv", index=False)
# df.to_csv("./labels.csv", index=False)
print('저장 완료')