import os
import os.path as osp
import time

from die_casting_loader import diecastingDataset
from model.die_casting_model import MultiLabelwithDensity
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
from configuration import BaseConfig, TrainConfig, TestConfig

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from sklearn.metrics import f1_score, recall_score, precision_score
# from torch.optim.lr_scheduler import CosineAnnealingLR


def train(cfg: BaseConfig, train_cfg: TrainConfig):
    writer = SummaryWriter(train_cfg.log_dir)
    
    # 폴더 존재 점검
    os.makedirs(train_cfg.model_dir, exist_ok=True)
    os.makedirs(train_cfg.log_dir, exist_ok=True)
    
    # 모델 정의
    model = MultiLabelwithDensity(num_classes=train_cfg.num_classes)
    model = model.to(cfg.device)
    
    # 학습 데이터
    dataset = diecastingDataset(cfg)
    train_loader = DataLoader(dataset, 
                              shuffle=True, 
                              batch_size=train_cfg.batch_size, 
                              pin_memory=True, 
                              num_workers=train_cfg.workers)
    
    # 최적화
    optimizer = torch.optim.Adam(model.parameters(), lr=train_cfg.lr)
    criterion = nn.BCEWithLogitsLoss()
    # scheduler = CosineAnnealingLR(optimizer, T_max=cfg.num_epochs)
    
    steps_per_epoch = len(train_loader)
    total_steps = steps_per_epoch * train_cfg.num_epochs
    
    for epoch in range(train_cfg.num_epochs):
        model.train()
        train_loss = torch.tensor(0., dtype=torch.float32, device=cfg.device)
        
        all_preds = []
        all_labels = []
        prev_f1 = 0.
        elapsed_time = 0.
        for idx, (image, label, density) in enumerate(train_loader):
            start = time.time()

            optimizer.zero_grad()
            
            image = image.to(cfg.device)
            label = label.to(cfg.device)
            density = density.to(cfg.device).unsqueeze(1)
            
            logits = model(image, density)

            loss = criterion(logits, label)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            
            preds = (torch.sigmoid(logits) > cfg.threshold).int().cpu().numpy()
            all_preds.append(preds)
            all_labels.append(label)
            
            end = time.time()
            elapsed_time += (end - start)
            if idx % 100 == 0:
                print_string = (f"Epoch: [{epoch + 1}/{cfg.num_epochs:>4d}] | Step: {idx:>5d}/{steps_per_epoch} | " 
                                f"Elapsed time: {elapsed_time/60:.3f}min | train_loss: {loss:>.4f}")
                print(print_string)
                elapsed_time = 0.

        avg_loss = train_loss / steps_per_epoch

        all_labels = torch.cat(all_labels).numpy()
        all_preds = torch.cat(all_preds).numpy()

        precision = precision_score(all_labels, all_preds, average='micro')
        recall = recall_score(all_labels, all_preds, average='micro')
        f1 = f1_score(all_labels, all_preds, average='micro')
        

        print(f'\nEpoch: [{epoch + 1}/{cfg.num_epochs:>5d}] | train_loss: {avg_loss:>.3f} | '
                f"Precision: {precision*100:.2f}% | Recall: {recall*100:.2f}% | F1: {f1*100:.2f}%")
        
        writer.add_scalar('optimization/train loss', avg_loss, global_step=epoch)
        writer.add_scalar("Metric/train precision", precision, global_step=epoch)
        writer.add_scalar("Metric/train recall", recall, global_step=epoch)
        writer.add_scalar("Metric/train f1", f1, global_step=epoch)

        # 모델 저장 코드
        if epoch == 1: prev_f1 = f1
        if epoch > 1 and f1 > prev_f1:
            torch.save(model.state_dict(), cfg.model_dir / f"{cfg.model_name}_{cfg.date_name}" / f"{epoch}.pth")
            

def test(test_cfg: TestConfig, cfg: TrainConfig, model):
    os.makedirs(test_cfg.test_results_dir, exist_ok=True)
    
    dataset = diecastingDataset(cfg.data_dir, mode='test')
    test_loader  = DataLoader(dataset, 
                                shuffle=True, 
                                batch_size=cfg.batch_size, 
                                pin_memory=True, 
                                num_workers=cfg.workers)
    
    model.load_state_dict(torch.load(osp.join(test_cfg.test_model_dir/f"{99}.pth"), map_location="cuda"))
    model.eval()
    
    all_labels = []
    all_preds = []
    avg_valid_loss = 0
    for idx, (image, label, density) in enumerate(test_loader):
        image = image.to(cfg.device)
        label = label.to(cfg.device)
        density = density.to(cfg.device).unsqueeze(1)
        
        with torch.no_grad():
            logits = model(image, density)
            valid_loss = criterion(logits, label)
        
        avg_valid_loss += valid_loss

        preds_prob = 1 / (1 + torch.exp(-logits))
        preds = (preds_prob > test_cfg.threshold).int()
        
        # 평가: F1-score, Average Precision(mAP), Recall
        y_true = label.cpu()
        y_pred = preds.cpu()
        
        all_labels.append(y_true)
        all_preds.append(y_pred)
    
    avg_valid_loss /= len(test_loader)

    y_true = torch.cat(all_labels).numpy()
    y_pred = torch.cat(all_preds).numpy()
    
    f1 = f1_score(y_true, y_pred, average="micro")
    recall = recall_score(y_true, y_pred, average="micro")
    precision = precision_score(y_true, y_pred, average="micro") 
    
    print_string = f"f1: {f1*100:>.2f}% | recall: {recall*100:>.2f}% | precision: {precision*100:>.2f}%"
    print(print_string)
        
def main():    
    cfg         = BaseConfig()
    train_cfg   = TrainConfig()
    test_cfg    = TestConfig()
    
    if cfg.model == 'train':
        train(cfg)
    
    elif cfg.model == 'test':
        test(test_cfg, cfg)
    
    
if __name__ == "__main__":
    main()