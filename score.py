from die_casting_loader import diecastingDataset
from losses.losses import SSIM

import torch
from torch.utils.data import DataLoader


def gram_matrix(F):
    """
    F: [B, C, H, W]
    return: [B, C, C]
    """
    B, C, H, W = F.shape
    F = F.view(B, C, H * W)
    G = torch.bmm(F, F.transpose(1, 2)) / (H * W)
    return G


def compoute_train_recon_score(data_cfg, train_cfg, model):
    dataset = diecastingDataset(data_cfg, mode='train')
    train_loader = DataLoader(dataset, 
                        shuffle=False, 
                        batch_size=train_cfg.batch_size, 
                        pin_memory=True, 
                        num_workers=train_cfg.workers)
    recon_scores = []
    with torch.no_grad():
        for img, _ in train_loader:
            img = img.cuda()
            _, pred = model(img)
            recon_score = compute_recon_score(pred, img)
            recon_scores.append(recon_score.cpu())

    recon_scores = torch.cat(recon_scores, dim=0)  # [N]
    recon_scores = recon_scores.numpy()
    
    return recon_scores


def compute_recon_score(pred, image):
    l1_score = torch.mean(torch.abs(image - pred), dim=(1,2,3))
    
    ssim = SSIM().cuda()
    ssim_map = ssim(pred, image)
    ssim_score = ssim_map.mean(dim=(1,2,3))

    recon_score = l1_score + 0.3 * ssim_score
    
    return recon_score


def compute_train_gram(data_cfg, train_cfg, model):
    dataset = diecastingDataset(data_cfg, mode='train')
    train_loader = DataLoader(dataset, 
                        shuffle=False, 
                        batch_size=train_cfg.batch_size, 
                        pin_memory=True, 
                        num_workers=train_cfg.workers)
    
    all_layer_grams = [[], [], []]
    with torch.no_grad():
        for img, _ in train_loader:
            img = img.cuda()
            
            _, feats = model.encoder(img)  # 중간 feature 사용!
            for i, f in enumerate(feats):
                all_layer_grams[i].append(gram_matrix(f).cpu())

    # 레이어별 평균 계산
    gram_means = []
    for layer_grams in all_layer_grams:
        combined = torch.cat(layer_grams, dim=0)
        gram_means.append(combined.mean(dim=0))
    
    return gram_means



def compute_train_gram(data_cfg, train_cfg, model):
    dataset = diecastingDataset(data_cfg, mode='train')
    train_loader = DataLoader(dataset, shuffle=False, batch_size=train_cfg.batch_size)
    
    # 레이어별로 평균 Gram Matrix를 저장할 리스트
    all_layer_grams = [[], [], []] # Layer 2, 3, 4 용
    
    with torch.no_grad():
        for img, _ in train_loader:
            img = img.cuda()
            _, feats = model.encoder(img) # 리스트 [f2, f3, f4] 수신
            
            for i, f in enumerate(feats):
                all_layer_grams[i].append(gram_matrix(f).cpu())

    # 레이어별 평균 계산
    gram_means = []
    for layer_grams in all_layer_grams:
        combined = torch.cat(layer_grams, dim=0)
        gram_means.append(combined.mean(dim=0)) # 각 레이어의 평균 Gram Matrix
    
    return gram_means # 리스트 반환


def compute_gram_distance(feats, gram_ref_means):
    total_dist = 0
    for i, f in enumerate(feats):
        G = gram_matrix(f)

        dist = torch.norm(G - gram_ref_means[i].cuda().unsqueeze(0), dim=(1,2))
        total_dist += dist / gram_ref_means[i].norm()
        
    return total_dist

