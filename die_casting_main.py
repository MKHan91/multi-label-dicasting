import os
import os.path as osp

from die_casting_loader import castingDataset
from die_casting_network import MultiLabelwithDensity
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR
import torch
import torch.nn as nn


def main():
    data_dir    = "/home/dev/DATASET/die_casting"
    model_dir   = osp.join(osp.dirname(__file__), 'experiments', 'models')
    log_dir     = osp.join(osp.dirname(__file__), 'experiments', 'logs')
    model_name  = 'resnet50'
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_epochs = 100
    mode = 'test'
    
    # 폴더 존재 점검
    if not osp.exists(model_dir):
        os.makedirs(model_dir)
    if not osp.exists(log_dir):
        os.makedirs(log_dir)
    
    # 텐서보드 셋팅
    writer = SummaryWriter(log_dir)
    
    
    model = MultiLabelwithDensity(num_classes=3)
    model = model.to(device)
    
    if mode == 'train':
        # 학습 데이터
        dataset = castingDataset(data_dir, mode='train')
        train_loader  = DataLoader(dataset, shuffle=True, batch_size=16, pin_memory=True, num_workers=8)
        
        # 최적화
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
        criterion = nn.BCEWithLogitsLoss()
        # scheduler = CosineAnnealingLR(optimizer, T_max=num_epochs)
        
        for epoch in range(num_epochs):
            model.train()
            train_loss = torch.tensor(0., dtype=torch.float32, device=device)
            
            for idx, (image, label, density) in enumerate(train_loader):
                optimizer.zero_grad()
                
                image = image.to(device)
                label = label.to(device)
                density = density.to(device).unsqueeze(1)
                
                logits = model(image, density)
                loss = criterion(logits, label)

                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
                
                if idx % 100 == 0:
                    print_string = (f"Epoch: [{epoch + 1}/{num_epochs:>4d}] | Step: {idx:>5d}/{len(train_loader)} | " 
                                    f"train_loss: {train_loss:>.4f}")
                    print(print_string)
                
            # scheduler.step()
            
            avg_loss = train_loss / len(train_loader)
            
            print('\n')
            print(f'Epoch: [{epoch + 1}/{num_epochs:>5d}] | train_loss: {avg_loss:>.3f}')
            
            writer.add_scalar('optimization/Train Loss', avg_loss, global_step=epoch)
            # 모델 저장 코드
            torch.save(model.state_dict(), osp.join(model_dir, model_name+'_'+epoch+f'{epoch}.pth'))
            

    elif mode == 'test':
        model.eval()
        
        threshold = 0.5
        for idx, (image, label, density) in enumerate(train_loader):
            image = image.to(device)
            label = label.to(device)
            density = density.to(device).unsqueeze(1)
            
            with torch.no_grad():
                # shape: (16, 3, 1)
                preds = model(image, density)

            # 0~1 사이의 확률 값
            # 예를들어, [[0.9, 0.9, 0.1], [0.9, 0.8, 0.95], [0.9, 0.8, 0.95], ..., [0.9, 0.8, 0.95]]
            preds = 1 / (1 + torch.exp(-preds))
            # [[True, True, False], [True, True, True]] --> [[1, 1, 0], [1, 1, 1]] 
            preds = (preds > threshold).int()
            
            print(f'정답 값: {label}')
            print(f'예측 값: {preds}')
            
            # 평가: F1-score, Average Precision(mAP), Recall
            
            
if __name__ == "__main__":
    main()