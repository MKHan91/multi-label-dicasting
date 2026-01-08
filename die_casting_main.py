import os
import os.path as osp
import time

from die_casting_loader import diecastingDataset
from model.die_casting_model import AnomalyDetector
from configuration import BaseConfig, DataConfig, TrainConfig, TestConfig
from losses import losses
import utils
import score

import seaborn as sns
import numpy as np
import pandas as pd
import shutil
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torchmetrics.classification import MultilabelF1Score, MultilabelPrecision, MultilabelRecall
from torchmetrics import MeanMetric
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader


# region - training
def train():
    loss_cfg = train_cfg.LossConfig()
    writer = SummaryWriter(train_cfg.log_dir)
    
    # 폴더 존재 점검
    os.makedirs(train_cfg.model_dir, exist_ok=True)
    os.makedirs(train_cfg.log_dir, exist_ok=True)
    os.makedirs(train_cfg.code_dir, exist_ok=True)
    utils.backup_codes(train_cfg.code_dir)
    
    # 모델 정의
    model = AnomalyDetector(train_cfg)
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
    ssim = losses.SSIM()
    ssim = ssim.to(device)
    
    poly_lambda = lambda epoch: (1 - epoch / train_cfg.num_epochs) ** 0.9
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=poly_lambda)
    # scheduler = CosineAnnealingLR(optimizer, T_max=train_cfg.num_epochs, eta_min=1e-6)

    metric_train_loss = MeanMetric().to(device)
    min_test_loss = float('inf')
    
    steps_per_epoch = len(train_loader)
    # total_steps = steps_per_epoch * train_cfg.num_epochs
    
    for epoch in range(train_cfg.num_epochs):
        model.train()
        metric_train_loss.reset() 
        
        for step, (image, image_name) in enumerate(train_loader):
            start = time.time()

            optimizer.zero_grad()
            image = image.to(device)
            
            z, pred = model(image)
            
            abs_diff = torch.abs(image - pred)
            l1_loss = abs_diff.mean()
            ssim_loss = ssim(pred, image).mean()
            
            recon_loss = loss_cfg.ssim_weight * ssim_loss + loss_cfg.l1_weight * l1_loss
            recon_loss.backward()
            optimizer.step()
            
            metric_train_loss.update(recon_loss)

            elapsed_time = (time.time() - start)
            if step % 1 == 0:
                print_string = (f"Model: [{train_cfg.model_name}] | Epoch: [{epoch + 1}/{train_cfg.num_epochs:>4d}] | Step: {step:>5d}/{steps_per_epoch} | " 
                                f"Elapsed time: {elapsed_time:.3f}sec | train_loss: {recon_loss:>.4f}")
                print(print_string)

        scheduler.step()
        
        avg_train_loss = metric_train_loss.compute().item()

        dataset_size = len(train_loader.dataset)
        random_indices = np.random.choice(dataset_size, 1, replace=False)
        image_sample, _ = train_loader.dataset[random_indices.item()]
        train_z, train_recon_image = model(image_sample.unsqueeze(0).to(device))
        
        test_loss, test_image_sample, test_recon_image = evaluate(model, loss_cfg)
        
        writer.add_scalar('optimization/learning_rate', optimizer.param_groups[0]['lr'], global_step=epoch)
        writer.add_scalar('optimization/train loss', avg_train_loss, global_step=epoch)
        writer.add_scalar('optimization/test loss', test_loss, global_step=epoch)
        writer.add_image('train_image/input', image_sample, global_step=epoch)
        writer.add_image('train_image/output', train_recon_image[0], global_step=epoch)
        writer.add_image('test_image/input', test_image_sample, global_step=epoch)
        writer.add_image('test_image/output', test_recon_image[0], global_step=epoch)

        # 모델 저장 코드
        if epoch > 50 and test_loss < min_test_loss:
            min_test_loss = test_loss
            torch.save(model.state_dict(), train_cfg.model_dir / f"{train_cfg.model_name}_{epoch:04d}.pth")
        elif epoch == train_cfg.num_epochs - 1:
            torch.save(model.state_dict(), train_cfg.model_dir / f"{train_cfg.model_name}_{epoch:04d}.pth")

# region - evaluation
def evaluate(model, loss_cfg):
    dataset = diecastingDataset(data_cfg, test_cfg, mode='test')
    test_loader = DataLoader(dataset, 
                             shuffle=True, 
                             batch_size=test_cfg.batch_size, 
                             pin_memory=True, 
                             num_workers=test_cfg.workers)
    model.eval()
    metric_test_loss = MeanMetric().to(device)
    
    print('------------------------ Start test ------------------------')
    # total_preds = torch.zeros(dataset.labels.shape, dtype=torch.uint8).cuda()
    # total_test_names = []
    ssim = losses.SSIM()
    ssim = ssim.to(device)
    
    for test_image, test_label, _ in test_loader:
        test_image = test_image.to(device)
        
        with torch.no_grad():
            z, pred = model(test_image)
            
            abs_diff = torch.abs(test_image - pred)
            l1_loss = abs_diff.mean()
            ssim_loss = ssim(pred, test_image).mean()
            
            recon_loss = loss_cfg.ssim_weight * ssim_loss + loss_cfg.l1_weight * l1_loss

        # if cfg.mode == "test":
        #     total_preds[idx*test_cfg.batch_size: (idx+1)*test_cfg.batch_size] = pred
        #     total_test_names.append(test_name)
            
        metric_test_loss.update(recon_loss)
    
    test_loss   = metric_test_loss.compute().item()
    
    dataset_size = len(test_loader.dataset)
    random_indices = np.random.choice(dataset_size, 1, replace=False)
    test_image_sample, _, _ = test_loader.dataset[random_indices.item()]
    test_z, test_recon_image = model(test_image_sample.unsqueeze(0).to(device))
    
    return test_loss, test_image_sample, test_recon_image


# region - inference
def inference(model):
    "visualize latent space with t-SNE"
    from sklearn.manifold import TSNE
    
    os.makedirs(test_cfg.latent_dir, exist_ok=True)
    
    gram_ref_means = score.compute_train_gram(data_cfg, train_cfg, model)
    
    dataset = diecastingDataset(data_cfg, test_cfg, mode='test')
    test_loader = DataLoader(dataset, 
                             shuffle=False, 
                             batch_size=test_cfg.batch_size, 
                             pin_memory=True, 
                             num_workers=test_cfg.workers)
    
    print('Exracting latent features...')
    # latent_list = []
    combined_features = [] # t-SNE에 사용할 결합 벡터
    test_labels = []
    
    model.eval()
    with torch.no_grad():
        for test_image, test_label, _ in test_loader:
            test_image = test_image.to(device)
            
            latent, feats = model.encoder(test_image)
            recon = model.decoder1(latent)
            
            gram_dist = score.compute_gram_distance(feats, gram_ref_means)
            feat_vec = torch.cat([latent, gram_dist.unsqueeze(1)], dim=1)
            
            # latent_list.append(latent.detach().cpu())
            combined_features.append(feat_vec.detach().cpu())
            test_labels.append(test_label.detach().cpu())
            
    # all_latents = torch.cat(latent_list, dim=0).numpy()
    all_latents = torch.cat(combined_features, dim=0).numpy()
    all_labels  = torch.cat(test_labels, dim=0)
    
    inv_test_class = {}
    for key, value in test_cfg.test_class_dict.items():
        inv_test_class[value] = key
    all_labels_names = [inv_test_class[label.item()] for label in all_labels]
    
    print('Applying t-SNE...')
    tsne = TSNE(n_components=2, random_state=42, perplexity=30, max_iter=300)
    latents_2d = tsne.fit_transform(all_latents)
    
    df = pd.DataFrame({
        'x': latents_2d[:,0],
        'y': latents_2d[:,1],
        'label': all_labels_names
    })
    
    plt.figure(figsize=(10, 7))
    sns.scatterplot(data=df, x='x', y='y', hue='label', palette='tab10', alpha=0.7)
    plt.title("t-SNE Visualization of Latent Space")
    plt.legend()
    plt.xlabel("feature 1")
    plt.ylabel("feature 2")
    saved_class_names = "_".join(test_cfg.target_classes)
    plt.savefig(test_cfg.latent_dir / f"{test_cfg.model_name}_{test_cfg.epoch:04d}_{saved_class_names}.png")
    plt.close()
    
    print('done')
    
    
def main():
    if cfg.mode == 'train':
        train()
    
    elif cfg.mode == 'test':
        model = AnomalyDetector(test_cfg)
        model = model.to(device)
        model.load_state_dict(torch.load(test_cfg.model_dir / f"{test_cfg.model_name}_{test_cfg.epoch:04d}.pth", map_location=device))
        
        inference(model)

    
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    cfg         = BaseConfig()
    data_cfg    = DataConfig()
    train_cfg   = TrainConfig()
    test_cfg    = TestConfig()
    
    main()