import os
import os.path as osp
import time

from die_casting_loader import diecastingDataset
from model.die_casting_model import MultiLabelwithDensity
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
from configuration import BaseConfig, DataConfig, TrainConfig, TestConfig

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torchmetrics.classification import MultilabelF1Score, MultilabelPrecision, MultilabelRecall
from torchmetrics import MeanMetric
# from torch.optim.lr_scheduler import CosineAnnealingLR


def train():
    writer = SummaryWriter(train_cfg.log_dir)
    
    # 폴더 존재 점검
    os.makedirs(train_cfg.model_dir, exist_ok=True)
    os.makedirs(train_cfg.log_dir, exist_ok=True)
    
    # 모델 정의
    model = MultiLabelwithDensity(train_cfg, num_classes=train_cfg.num_classes)
    model = model.to(device)
    
    # 학습 데이터
    dataset = diecastingDataset(data_cfg, mode='train')
    train_loader = DataLoader(dataset, 
                              shuffle=True, 
                              batch_size=train_cfg.batch_size, 
                              pin_memory=True, 
                              num_workers=train_cfg.workers)
    
    # 최적화
    optimizer = torch.optim.Adam(model.parameters(), lr=train_cfg.lr)
    criterion = nn.BCEWithLogitsLoss()
    # scheduler = CosineAnnealingLR(optimizer, T_max=cfg.num_epochs)
    
    metric_recall = MultilabelRecall(num_labels=len(data_cfg.label_list), average='micro').to(device)
    metric_precision = MultilabelPrecision(num_labels=len(data_cfg.label_list), average='micro').to(device)
    metric_f1 = MultilabelF1Score(num_labels=len(data_cfg.label_list), average='micro').to(device)
    metric_train_loss = MeanMetric().to(device)
    
    
    steps_per_epoch = len(train_loader)
    total_steps = steps_per_epoch * train_cfg.num_epochs
    for epoch in range(train_cfg.num_epochs):
        model.train()
        
        metric_recall.reset()
        metric_precision.reset()
        metric_f1.reset()
        metric_train_loss.reset()
        
        # train_loss = torch.tensor(0., dtype=torch.float32, device=device)
        prev_f1 = torch.tensor(0., dtype=torch.float32, device=device)
        for idx, (image, label, density) in enumerate(train_loader):
            start = time.time()

            optimizer.zero_grad()
            
            image = image.to(device)
            label = label.to(device)
            density = density.to(device).unsqueeze(1)
            
            logits = model(image, density)

            loss = criterion(logits, label)
            loss.backward()
            optimizer.step()
            
            # train_loss += loss.item()
            metric_train_loss.update(loss)
            
            preds = (torch.sigmoid(logits) > train_cfg.train_thld).int()
            metric_recall.update(preds, label)
            metric_precision.update(preds, label)
            metric_f1.update(preds, label)

            elapsed_time = (time.time() - start)
            if idx % 100 == 0:
                print_string = (f"Epoch: [{epoch + 1}/{train_cfg.num_epochs:>4d}] | Step: {idx:>5d}/{steps_per_epoch} | " 
                                f"Elapsed time: {elapsed_time:.3f}sec | train_loss: {loss:>.4f}")
                print(print_string)

        avg_train_loss = metric_train_loss.compute().item()
        train_recall = metric_recall.compute()
        train_precision = metric_precision.compute()
        train_f1 = metric_f1.compute()
        
        test_loss, test_f1, test_precision, test_recall = test(model, loss_fn=criterion)
        print(f'\nEpoch: [{epoch + 1}/{cfg.num_epochs:>5d}] | test_loss: {test_loss:>.3f} | '
              f"Test Precision: {test_precision*100:.2f}% | Test Recall: {test_recall*100:.2f}% | Test F1: {test_f1*100:.2f}%")
        
        writer.add_scalar('optimization/train loss', avg_train_loss, global_step=epoch)
        writer.add_scalar("Metric/train precision", train_precision, global_step=epoch)
        writer.add_scalar("Metric/train recall", train_recall, global_step=epoch)
        writer.add_scalar("Metric/train f1", train_f1, global_step=epoch)
        writer.add_scalar("Metric/test precision", test_precision, global_step=epoch)
        writer.add_scalar("Metric/test recall", test_recall, global_step=epoch)
        writer.add_scalar("Metric/test f1", test_f1, global_step=epoch)

        # 모델 저장 코드
        if epoch == 1: prev_f1 = test_f1
        if epoch > 1 and test_f1 > prev_f1:
            torch.save(model.state_dict(), test_cfg.save_dir / f"{test_cfg.test_model_name}_{epoch}.pth")
            

def test(model, loss_fn):
    os.makedirs(test_cfg.test_results_dir, exist_ok=True)
    
    dataset = diecastingDataset(data_cfg, mode='test')
    test_loader = DataLoader(dataset, 
                             shuffle=True, 
                             batch_size=cfg.batch_size, 
                             pin_memory=True, 
                             num_workers=cfg.workers)
    if test_cfg.do_infer:
        model.load_state_dict(torch.load(osp.join(test_cfg.test_model_dir/f"{99}.pth"), map_location="cuda"))
    
    model.eval()
    
    metric_recall = MultilabelRecall(num_labels=len(data_cfg.label_list), average='micro').to(device)
    metric_precision = MultilabelPrecision(num_labels=len(data_cfg.label_list), average='micro').to(device)
    metric_f1 = MultilabelF1Score(num_labels=len(data_cfg.label_list), average='micro').to(device)
    metric_test_loss = MeanMetric().to(device)
    
    for image, label, density in test_loader:
        image = image.to(device)
        label = label.to(device)
        density = density.to(device).unsqueeze(1)
        
        with torch.no_grad():
            logits = model(image, density)
            test_loss = loss_fn(logits, label)

        preds_prob = 1 / (1 + torch.exp(-logits))
        preds = (preds_prob > test_cfg.threshold).int()

        metric_test_loss.update(test_loss)
        metric_recall.update(preds, label)
        metric_precision.update(preds, label)
        metric_f1.update(preds, label)
    
    recall      = metric_recall.compute().item()
    precision   = metric_precision.compute().item()
    f1          = metric_f1.compute().item()
    test_loss   = metric_test_loss.compute().item()
    
    metric_recall.reset()
    metric_precision.reset()
    metric_f1.reset()
    metric_recall.reset()
    
    return test_loss, f1, precision, recall

def main():    
    if cfg.mode == 'train':
        train()
    
    elif cfg.mode == 'test':
        test()
    
    
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    cfg         = BaseConfig()
    data_cfg    = DataConfig()
    train_cfg   = TrainConfig()
    test_cfg    = TestConfig()
    
    main()