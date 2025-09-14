import os
import os.path as osp

from die_casting_loader import castingDataset
from die_casting_network import MultiLabelwithDensity
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
from configuration import TrainConfig, TestConfig

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from sklearn.metrics import f1_score, recall_score, precision_score
# from torch.optim.lr_scheduler import CosineAnnealingLR

def main():
    cfg = TrainConfig()
    
    model = MultiLabelwithDensity(num_classes=cfg.num_classes)
    model = model.to(cfg.device)
    
    if cfg.mode == 'train':
        # 폴더 존재 점검
        os.makedirs(cfg.model_dir, exist_ok=True)
        os.makedirs(cfg.log_dir, exist_ok=True)
        
        writer = SummaryWriter(cfg.log_dir)
        
        # 학습 데이터
        dataset = castingDataset(cfg.data_dir, mode='train')
        train_loader  = DataLoader(dataset, 
                                   shuffle=True, 
                                   batch_size=cfg.batch_size, 
                                   pin_memory=True, 
                                   num_workers=cfg.workers)
        
        # 최적화
        optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)
        criterion = nn.BCEWithLogitsLoss()
        # scheduler = CosineAnnealingLR(optimizer, T_max=cfg.num_epochs)
        
        for epoch in range(cfg.num_epochs):
            model.train()
            train_loss = torch.tensor(0., dtype=torch.float32, device=cfg.device)
            
            for idx, (image, label, density) in enumerate(train_loader):
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
                    print_string = (f"\rEpoch: [{epoch + 1}/{cfg.num_epochs:>4d}] | Step: {idx:>5d}/{len(train_loader)} | " 
                                    f"train_loss: {train_loss:>.4f}")
                    print(print_string, end='')
                
            # scheduler.step()
            
            avg_loss = train_loss / len(train_loader)
            
            print('\n')
            print(f'Epoch: [{epoch + 1}/{cfg.num_epochs:>5d}] | train_loss: {avg_loss:>.3f}')
            
            writer.add_scalar('optimization/Train Loss', avg_loss, global_step=epoch)
            # 모델 저장 코드
            if epoch % 10 == 0:
                torch.save(model.state_dict(), cfg.model_dir / f"{cfg.model_name}_{cfg.date_name}" / f"{epoch}.pth")
            

    elif cfg.mode == 'test':
        test_cfg = TestConfig()
        
        os.makedirs(test_cfg.test_results_dir, exist_ok=True)
        
        dataset = castingDataset(cfg.data_dir, mode='test')
        test_loader  = DataLoader(dataset, 
                                  shuffle=True, 
                                  batch_size=cfg.batch_size, 
                                  pin_memory=True, 
                                  num_workers=cfg.workers)
        
        model.load_state_dict(torch.load(osp.join(test_cfg.test_model_dir/f"{99}.pth"), map_location="cuda"))
        model.eval()
        
        all_labels = []
        all_preds = []
        for idx, (image, label, density) in enumerate(test_loader):
            image = image.to(cfg.device)
            label = label.to(cfg.device)
            density = density.to(cfg.device).unsqueeze(1)
            
            with torch.no_grad():
                preds = model(image, density)

            preds_prob = 1 / (1 + torch.exp(-preds))
            preds = (preds_prob > test_cfg.threshold).int()
            
            # 평가: F1-score, Average Precision(mAP), Recall
            y_true = label.cpu()
            y_pred = preds.cpu()
            
            all_labels.append(y_true)
            all_preds.append(y_pred)
            
        y_true = torch.cat(all_labels).numpy()
        y_pred = torch.cat(all_preds).numpy()
        
        f1 = f1_score(y_true, y_pred, average="micro")
        recall = recall_score(y_true, y_pred, average="micro")
        precision = precision_score(y_true, y_pred, average="micro") 
        
        print_string = f"f1: {f1*100:>.2f}% | recall: {recall*100:>.2f}% | precision: {precision*100:>.2f}%"
        print(print_string)
        
        
if __name__ == "__main__":
    main()