import os
import os.path as osp

from die_casting_loader import castingDataset
from die_casting_network import MultiLabelwithDensity
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR
import torch
import torch.nn as nn

from configuration import TrainConfig 



def main():
    cfg = TrainConfig()
    
    # 폴더 존재 점검
    os.makedirs(cfg.model_dir / cfg.model_name, exist_ok=True)
    os.makedirs(cfg.log_dir, exist_ok=True)
    
    # 텐서보드 셋팅
    writer = SummaryWriter(cfg.log_dir)
    
    model = MultiLabelwithDensity(num_classes=3)
    model = model.to(cfg.device)
    
    if cfg.mode == 'train':
        # 학습 데이터
        dataset = castingDataset(cfg.data_dir, mode='train')
        train_loader  = DataLoader(dataset, shuffle=True, batch_size=16, pin_memory=True, num_workers=8)
        
        # 최적화
        optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)
        criterion = nn.BCEWithLogitsLoss()
        # scheduler = CosineAnnealingLR(optimizer, T_max=cfg.num_epochs)
        
        for epoch in range(cfg.num_epochs):
            model.train()
            train_loss = torch.tensor(0., dtype=torch.float32, device=cfg.device)
            
            for idx, (image, label, density) in enumerate(train_loader):
                if idx != 1: break
                optimizer.zero_grad()
                
                image = image.to(cfg.device)
                label = label.to(cfg.device)
                density = density.to(cfg.device).unsqueeze(1)
                
                logits = model(image, density)
                loss = criterion(logits, label)

                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
                
                if idx % 100 == 0:
                    print_string = (f"Epoch: [{epoch + 1}/{cfg.num_epochs:>4d}] | Step: {idx:>5d}/{len(train_loader)} | " 
                                    f"train_loss: {train_loss:>.4f}")
                    print(print_string)
                
            # scheduler.step()
            
            avg_loss = train_loss / len(train_loader)
            
            print('\n')
            print(f'Epoch: [{epoch + 1}/{cfg.num_epochs:>5d}] | train_loss: {avg_loss:>.3f}')
            
            writer.add_scalar('optimization/Train Loss', avg_loss, global_step=epoch)
            # 모델 저장 코드
            if epoch > 0:
                torch.save(model.state_dict(), cfg.model_dir / cfg.model_name / f"{epoch}.pth")
            

    elif cfg.mode == 'test':
        model.eval()
        
        threshold = 0.5
        for idx, (image, label, density) in enumerate(train_loader):
            image = image.to(cfg.device)
            label = label.to(cfg.device)
            density = density.to(cfg.device).unsqueeze(1)
            
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