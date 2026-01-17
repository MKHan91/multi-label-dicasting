from die_casting_loader import diecastingDataset
from losses.losses import SSIM

import numpy as np
import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F


def gram_matrix(F):
    B, C, H, W = F.shape
    F = F.view(B, C, H * W)
    # H*W: 픽셀 개수에 따른 스케일 보정
    # C*H*W: 채널 개수에 따른 에너지 총합의 팽창을 막기 위함. 
    G = torch.bmm(F, F.transpose(1, 2)) / (C * H * W)
    return G

class AnomalyScore:
    def __init__(self, dcfg, tcfg, test_cfg, device='cpu'):
        self.dcfg = dcfg
        self.tcfg = tcfg
        self.test_cfg = test_cfg
        self.device = device
        

    def compute_train_recon(self, model):
        dataset = diecastingDataset(self.dcfg, mode='train')
        train_loader = DataLoader(dataset, 
                            shuffle=False, 
                            batch_size=self.tcfg.batch_size, 
                            pin_memory=True, 
                            num_workers=self.tcfg.workers)
        recon_scores = []
        with torch.no_grad():
            for img, _ in train_loader:
                img = img.to(self.device)
                _, pred = model(img)
                recon_score = self.compute_recon_score(self.tcfg, pred, img)
                recon_scores.append(recon_score.cpu())

        recon_scores = torch.cat(recon_scores, dim=0)  # [N]
        recon_scores = recon_scores.numpy()
        
        return recon_scores

    def compute_train_gram(self, model):
        dataset = diecastingDataset(self.dcfg, mode='train')
        train_loader = DataLoader(dataset, 
                            shuffle=False, 
                            batch_size=self.tcfg.batch_size, 
                            pin_memory=True, 
                            num_workers=self.tcfg.workers)
        
        all_layer_grams = [[], [], []]
        with torch.no_grad():
            for img, _ in train_loader:
                img = img.to(self.device)
                
                _, feats = model.encoder1(img)
                for i, f in enumerate(feats):
                    gram = gram_matrix(f)
                    gram_vec = gram.view(gram.size(0), -1)
                    gram_vec = F.normalize(gram_vec, p=2, dim=1)

                    all_layer_grams[i].append(gram_vec.cpu())
                    # all_layer_grams[i].append(gram_matrix(f).cpu())

        gram_means = []
        for layer_grams in all_layer_grams:
            combined = torch.cat(layer_grams, dim=0)
            # gram_means.append(combined.mean(dim=0))
            cmb_vec = combined.mean(dim=0, keepdim=True)
            norm_cmb_vec = F.normalize(cmb_vec, p=2, dim=1)
            gram_means.append(norm_cmb_vec.to(self.device))
        
        return gram_means
    

    def compute_recon_score(self, pred, image):
        l1_score = torch.mean(torch.abs(image - pred), dim=(1,2,3))
        
        ssim = SSIM().to(self.device)
        ssim_map = ssim(pred, image)
        ssim_score = ssim_map.mean(dim=(1,2,3))

        recon_score = self.test_cfg.test_l1_weight * l1_score + self.test_cfg.test_ssim_weight * ssim_score
        
        return recon_score


    def compute_cosine_similarity(self, feats, gram_ref_means):
        # total_dist = 0
        # similarity = torch.zeros(feats[0].shape[0], len(feats))
        similarity = []
        for i, f in enumerate(feats):
            gram = gram_matrix(f)
            gram_vec = gram.view(gram.size(0), -1)
            gram_vec = F.normalize(gram_vec, p=2, dim=1)
            
            cosine_sim = torch.mm(gram_vec, gram_ref_means[i].t())
            # similarity[:, i] += cosine_sim[:, 0]
            similarity.append(cosine_sim)
            # dist = torch.norm(gram - gram_ref_means[i].to(self.device).unsqueeze(0), dim=(1,2))
            # total_dist += dist / gram_ref_means[i].norm()
            
        # return total_dist
        similarity = torch.cat(similarity, dim=1)
        return similarity.mean(dim=1)


    # region - anomaly score
    def compute_final_anomaly_score(self, pred, image, feats, gram_ref_means):
        recon_score = self.compute_recon_score(pred, image)
        gram_score = self.compute_cosine_similarity(feats, gram_ref_means)
        
        # final_score = recon_score + 0.5 * gram_score
        final_score = 0.5 * recon_score + (1. - gram_score)
        
        return final_score


    def find_threshold_by_density(self, scores, alpha=6):
        from scipy.stats import gaussian_kde
        import matplotlib.pyplot as plt
        import seaborn as sns
        
        
        kde = gaussian_kde(scores)
        x_range = np.linspace(min(scores), max(scores), 1000)
        density = kde(x_range)
        
        max_density_idx = np.argmax(density)
        most_common_score = x_range[max_density_idx]
        
        std_dev = np.std(scores)
        threshold = most_common_score + (alpha * std_dev)
        
        sns.histplot(scores, kde=True, color='green', label='Train (Normal)', stat="density", bins=50)
        plt.savefig(self.test_cfg.figure_dir / f"{self.test_cfg.model_name}_{self.test_cfg.epoch:04d}_density.png")
        plt.close()
        
        return most_common_score, threshold
    
    
def get_threshold_from_train(data_cfg, 
                             train_cfg, 
                             model, 
                             gram_ref_means, 
                             percentile=95):
    
    dataset = diecastingDataset(data_cfg, mode='train')
    train_loader = DataLoader(dataset, 
                        shuffle=False, 
                        batch_size=train_cfg.batch_size, 
                        pin_memory=True, 
                        num_workers=train_cfg.workers)
    train_scores = []
    with torch.no_grad():
        for img, _ in train_loader:
            img = img.to(self.device)
            
            latent, feats = model.encoder1(img)
            recon = model.decoder1(latent)
            
            score = compute_final_anomaly_score(recon, img, feats, gram_ref_means)
            train_scores.append(score.detach().cpu())
        
    train_scores=  torch.cat(train_scores).numpy()
    # 학습 데이터(정상) 점수 중 95% 지점의 점수를 임계치로 설정
    threshold = np.percentile(train_scores, percentile)
    
    return threshold