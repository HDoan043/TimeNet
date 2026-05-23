# This source code is provided for the purposes of scientific reproducibility
# under the following limited license from Element AI Inc. The code is an
# implementation of the N-BEATS model (Oreshkin et al., N-BEATS: Neural basis
# expansion analysis for interpretable time series forecasting,
# https://arxiv.org/abs/1905.10437). The copyright to the source code is
# licensed under the Creative Commons - Attribution-NonCommercial 4.0
# International license (CC BY-NC 4.0):
# https://creativecommons.org/licenses/by-nc/4.0/.  Any commercial use (whether
# for the benefit of third parties or internally in production) requires an
# explicit license. The subject-matter of the N-BEATS model and associated
# materials are the property of Element AI Inc. and may be subject to patent
# protection. No license to patents is granted hereunder (whether express or
# implied). Copyright © 2020 Element AI Inc. All rights reserved.

"""
Loss functions for PyTorch.
"""

import torch as t
import torch.nn as nn
import numpy as np
import pdb

def gaussian_blur_1d(labels, kernel_size=5, sigma=1.0):
        """
        labels: [B, W]
        return: [B, W]
        """
        x = t.arange(kernel_size, device=labels.device) - kernel_size // 2
        kernel = t.exp(-(x**2) / (2 * sigma**2))
        kernel = kernel / kernel.sum()
        # reshape cho conv1d
        kernel = kernel.view(1, 1, kernel_size)
        
        # input shape: [B, 1, W]
        labels = labels.unsqueeze(1)
        padding = kernel_size // 2
        
        blurred = nn.functional.conv1d( labels, kernel, padding=padding)
        
        return blurred.squeeze(1)
    
def divide_no_nan(a, b):
    """
    a/b where the resulted NaN or Inf are replaced by 0.
    """
    result = a / b
    result[result != result] = .0
    result[result == np.inf] = .0
    return result


class mape_loss(nn.Module):
    def __init__(self):
        super(mape_loss, self).__init__()

    def forward(self, insample: t.Tensor, freq: int,
                forecast: t.Tensor, target: t.Tensor, mask: t.Tensor) -> t.float:
        """
        MAPE loss as defined in: https://en.wikipedia.org/wiki/Mean_absolute_percentage_error

        :param forecast: Forecast values. Shape: batch, time
        :param target: Target values. Shape: batch, time
        :param mask: 0/1 mask. Shape: batch, time
        :return: Loss value
        """
        weights = divide_no_nan(mask, target)
        return t.mean(t.abs((forecast - target) * weights))


class smape_loss(nn.Module):
    def __init__(self):
        super(smape_loss, self).__init__()

    def forward(self, insample: t.Tensor, freq: int,
                forecast: t.Tensor, target: t.Tensor, mask: t.Tensor) -> t.float:
        """
        sMAPE loss as defined in https://robjhyndman.com/hyndsight/smape/ (Makridakis 1993)

        :param forecast: Forecast values. Shape: batch, time
        :param target: Target values. Shape: batch, time
        :param mask: 0/1 mask. Shape: batch, time
        :return: Loss value
        """
        return 200 * t.mean(divide_no_nan(t.abs(forecast - target),
                                          t.abs(forecast.data) + t.abs(target.data)) * mask)


class mase_loss(nn.Module):
    def __init__(self):
        super(mase_loss, self).__init__()

    def forward(self, insample: t.Tensor, freq: int,
                forecast: t.Tensor, target: t.Tensor, mask: t.Tensor) -> t.float:
        """
        MASE loss as defined in "Scaled Errors" https://robjhyndman.com/papers/mase.pdf

        :param insample: Insample values. Shape: batch, time_i
        :param freq: Frequency value
        :param forecast: Forecast values. Shape: batch, time_o
        :param target: Target values. Shape: batch, time_o
        :param mask: 0/1 mask. Shape: batch, time_o
        :return: Loss value
        """
        masep = t.mean(t.abs(insample[:, freq:] - insample[:, :-freq]), dim=1)
        masked_masep_inv = divide_no_nan(mask, masep[:, None])
        return t.mean(t.abs(target - forecast) * masked_masep_inv)

class NTXentLoss(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.temperature = args.temperature
        self.mse = nn.MSELoss()
        self.neighbor_sim_anchor = min(args.neighbor_sim_anchor, args.batch_size-1)
        self.neighbor_sim_pos = min(args.neighbor_sim_pos, args.batch_size-1)
        self.emphasize_negative = args.emphasize_negative
        self.reconstruct_weight = args.reconstruct_weight
        self.contrastive_weight = args.contrastive_weight

    def forward(self, x, x_hat, z, idx, pos_idx, neg_idx, labels, attn_pooling, base_mse):
        # same device
        idx = idx.to(x.device)
        pos_idx = pos_idx.to(x.device)
        neg_idx = neg_idx.to(x.device)
        attn_pooling = attn_pooling.to(x.device)
        labels = labels.to(x.device)

        # reconstruct loss (anchor only)
        B = idx.shape[0]
        recon_loss = self.mse(x[idx], x_hat[idx])

        # normalize
        # collapse
        # -------------- MEAN ---------------------
        # z = z.mean(dim=1)                                                    # z: [3*B, 1, d_model]
        # -------------- MAX POOLING ----------------
        # z = z.max(dim=1).values                                             # z: [3B, 1, d_model]
        # -------------- LAST TIMESTAMP ------------------
        # z = z[:, -1, :]
        # -------------- ATTENTION POOLING -------------------
        # z = (z*attn_pooling).sum(dim=1)                                      # z: [3B, 1, d_model]
        # z = nn.functional.normalize(z, dim=1)                                     # z: [3B, 1, d_model]
        # -------------- ATTENTION POOLING WITH EMPHASIZED ANOMALY ----------------------
        labels = gaussian_blur_1d(labels)                                      # labels: [B, win_size]
        labels = labels.unsqueeze(-1)                                          # labels: [B, win_size, 1]
        alpha = self.args.label_guided_weight
        neg_att = attn_pooling[neg_idx]
        neg_att = neg_att*(1+ alpha*labels)                           
        neg_att = neg_att/(neg_att.sum(dim=1, keepdim=True))    

        attn_anchor = attn_pooling[idx]
        attn_pos = attn_pooling[pos_idx]
        z_anchor = (z[idx] * attn_anchor).sum(dim=1)
        z_pos = (z[pos_idx] * attn_pos).sum(dim=1)
        z_neg = (z[neg_idx] * neg_att).sum(dim=1)
        z = t.cat([z_anchor, z_pos, z_neg], dim=0)
        z = nn.functional.normalize(z, dim=1)                                   # z: [3B, 1, d_model]
        
        # similarity matrix (3B x 3B)
        sim = t.matmul(z, z.T) / self.temperature                        # sim: [3B, 3B]
        
        # mask self similarity ( all similarity between the representation of a sample and itself are ignored)
        mask = t.eye(sim.shape[0], device=sim.device).bool()
        sim.masked_fill_(mask, -t.inf)

        # positive similarity
        # positive samples of an anchor are the windows near the anchor (distance from the anchor is small enough) and their augmentations
        r = np.random.rand()
        # if r < 0.8: neighbor_sim_anchor = self.neighbor_sim_anchor
        # else: neighbor_sim_anchor = 0
        neighbor_sim_anchor = self.neighbor_sim_anchor
        pos_anchor_mask = t.zeros((B,B), device=sim.device)                      # pos_anchor_mask: [B, B]
        for i in range(1, neighbor_sim_anchor+1):
            pos_anchor_mask.diagonal(offset=i).fill_(1)
            pos_anchor_mask.diagonal(offset=-i).fill_(1)
        pos_mask = t.eye(B, device=sim.device)                                   # pos_mask: [B, B]
        for i in range(1, self.neighbor_sim_pos+1):
            pos_mask.diagonal(offset=i).fill_(1)
            pos_mask.diagonal(offset=-i).fill_(1)
        neg_mask = t.zeros((B,B), device=sim.device)                             # neg_mask: [B, B]
        full_pos_mask = t.cat([pos_anchor_mask, pos_mask, neg_mask], dim=1)      # full_pos_mask: [B, 3B], full_pos_mask[i,j] = 1 if sample[j] is a positive sample of anchor[i], = 0 else
        full_pos_mask = full_pos_mask.bool()
        logits = sim - sim.max(dim=1, keepdim=True)[0]
        exp_sim = t.exp(logits)
        pos_exp = exp_sim[idx] * full_pos_mask
        pos_sum = pos_exp.sum(dim=1)    
        
        # denominator
        full_neg_mask = ~full_pos_mask
        full_neg_mask.diagonal(offset=0).fill_(0)
        weights = t.ones_like(exp_sim[idx], device=exp_sim.device)               # weights: [B, 3B]
        weights[idx, neg_idx] += self.emphasize_negative
        
        # chọn top-k hardest negatives
        # k = int(0.1 * weights.shape[1])
        # hard_neg_mask = torch.zeros_like(weights)
        # topk_idx = torch.topk(neg_sim, k=k, dim=1).indices
        
        # hard_neg_mask.scatter_(1, topk_idx, 1)
        
        # weights = weights * (1 + alpha * hard_neg_mask)
        # weights = weights / weights.sum(dim=1, keepdim=True)
        
        denom = (exp_sim[idx]*weights).sum(dim=1)                                    # denom: [B]

        loss = -t.log(pos_sum / denom)
        
        # print(f"reconstruct_loss: {recon_loss.item()}")
        # print(f"NT-Xent Loss: {contrastive_weight*loss.mean().item()}")
        # ------------------ ADAPTIVE WEIGHT -----------------
        # contrastive_weight = recon_loss.detach()/( loss.mean().detach() + 1e-6)
        # contrastive_weight = contrastive_weight.clamp(0.1,10)
        # ------------------- FIXED WEIGHT -------------------
        contrastive_weight = self.contrastive_weight
        reconstruct_weight = self.args.reconstruct_weight
        return reconstruct_weight*recon_loss + contrastive_weight*loss.mean()
        
class TripletLoss(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.reconstruct_weight = args.reconstruct_weight
        self.contrastive_weight = args.contrastive_weight
        self.triplet = nn.TripletMarginLoss(margin=args.margin, p=2)
        self.mse = nn.MSELoss()
    def forward(self, x, x_hat, z, idx, pos_idx, neg_idx, labels, attn_pooling, base_mse):
        # same device
        idx = idx.to(x.device)
        pos_idx = pos_idx.to(x.device)
        neg_idx = neg_idx.to(x.device)
        attn_pooling = attn_pooling.to(x.device)
        labels = labels.to(x.device)
        
        # reconstruct loss (anchor only)
        B = idx.shape[0]
        recon_loss = self.mse(x[idx], x_hat[idx])

        # normalize
        # -------------- ATTENTION POOLING -------------------
        # z = (z*attn_pooling).sum(dim=1)                                      # z: [3B, 1, d_model]
        # z = nn.functional.normalize(z, dim=1)                                # z: [3*batch_size, 1, d_model]
        # -------------- ATTENTION POOLING WITH EMPHASIZED ANOMALY ----------------------
        labels = gaussian_blur_1d(labels)                                 # labels: [B, win_size]
        labels = labels.unsqueeze(-1)                                          # labels: [B, win_size, 1]
        alpha = self.args.label_guided_weight
        neg_att = attn_pooling[neg_idx]
        neg_att = neg_att*(1+ alpha*labels)                           
        neg_att = neg_att/(neg_att.sum(dim=1, keepdim=True))    

        attn_anchor = attn_pooling[idx]
        attn_pos = attn_pooling[pos_idx]
        z_anchor = (z[idx] * attn_anchor).sum(dim=1)
        z_pos = (z[pos_idx] * attn_pos).sum(dim=1)
        z_neg = (z[neg_idx] * neg_att).sum(dim=1)
        z = t.cat([z_anchor, z_pos, z_neg], dim=0)
        z = nn.functional.normalize(z, dim=1)

        z_anchor = z[idx]                                                    # z_anchor: [batch_sze, d_model]
        z_pos = z[pos_idx]                                                   # z_pos: [batch_size, d_model]
        z_neg = z[neg_idx]                                                   # z_neg: [batch_size, d_model]

        loss = self.triplet(z_anchor, z_pos, z_neg)
        reconstruct_weight = self.reconstruct_weight
        contrastive_weight = self.contrastive_weight
        return reconstruct_weight*recon_loss + contrastive_weight*loss

class SeSimiLoss(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.margin = args.margin
        self.reconstruct_weight = args.reconstruct_weight
        self.contrastive_weight = args.contrastive_weight
        self.hard_mask = args.hard_mask
        self.max_ratio = getattr(args, 'max_ratio', 1.0)
        self.pos_ratio = getattr(args, 'pos_ratio', 0.5)
        
        self.mse_none = nn.MSELoss(reduction='none')

        self.temperature = getattr(args, 'temperature', 0.5)
        self.alpha_floor = getattr(args, 'alpha_floor', 0.2)
        
        self.recon_tolerance = getattr(args, 'recon_tolerance', 0.2) 
        self.throttle_beta = getattr(args, 'throttle_beta', 3.0) 

        self.magnitude_mode = getattr(args, 'magnitude_mode', 'variance')
        
    def forward(self, x, x_hat, z, idx, pos_idx, neg_idx, labels, attn_pooling, base_mse):
        if t.isnan(z).any() or t.isnan(x_hat).any():
            print("\n[BÁO ĐỘNG ĐỎ]: Đầu vào z hoặc x_hat đã bị NaN từ mô hình TimesNet TRƯỚC KHI tính Loss!")
            print(f"Có NaN ở z: {t.isnan(z).any().item()} | Có NaN ở x_hat: {t.isnan(x_hat).any().item()}")
        idx = idx.to(x.device)
        pos_idx = pos_idx.to(x.device)
        neg_idx = neg_idx.to(x.device)
        attn_pooling = attn_pooling.to(x.device)
        labels = labels.to(x.device)
        batch_base_mse = base_mse.mean().to(x.device).float()
        B = idx.shape[0]

        # ==========================================
        # 0. RECONSTRUCT LOSS
        # ==========================================
        sample_recon = self.mse_none(x[idx], x_hat[idx]).mean(dim=(1,2))             # [B]
        raw_recon_loss = sample_recon.mean()                                         # [1]

        if B > 1:
            batch_base_stat = t.quantile(base_mse, 0.8)                                # [1]
            batch_recon_stat = t.quantile(sample_recon.detach(), 0.8)                  # [1]
        else:
            batch_base_stat = base_mse.squeeze()
            batch_recon_stat = sample_recon.detach().squeeze()
                
        batch_anchor = z[idx]                                                              # [B, win_size, d_model]
        batch_positive = z[pos_idx]                                                        # [B, win_size, d_model]
        batch_negative = z[neg_idx]                                                        # [B, win_size, d_model]
        
        batch_anchor_norm = nn.functional.normalize(batch_anchor, p=2, dim=-1, eps=1e-5).float()
        batch_positive_norm = nn.functional.normalize(batch_positive, p=2, dim=-1, eps=1e-5).float()
        batch_negative_norm = nn.functional.normalize(batch_negative, p=2, dim=-1, eps=1e-5).float()
        
        try:
            blur_label = gaussian_blur_1d(labels).unsqueeze(-1)                            # [B, win_size, 1]
        except NameError:
            blur_label = labels.unsqueeze(-1)                                              # [B, win_size, 1]

        hard_label = labels.unsqueeze(-1)                                                  # [B, win_size, 1]
        
        hard_mask = self.max_ratio*t.maximum(hard_label, hard_label.transpose(1,2))\
             + (1-self.max_ratio)*t.matmul(hard_label, hard_label.transpose(1,2))          # [B, win_size, win_size]
        soft_mask = self.max_ratio*t.maximum(blur_label, blur_label.transpose(1,2))\
             + (1-self.max_ratio)*t.matmul(blur_label, blur_label.transpose(1,2))          # [B, win_size, win_size]
        mask = hard_mask if self.hard_mask == 1 else soft_mask                             # [B, win_size, win_size]
        
        anomaly_element = mask.sum(dim=(1,2))                                              # [B]
        normal_element = (1-mask).sum(dim=(1,2))                                           # [B]
            
        # ==========================================
        # 1. SIMILARITY BRANCH (Dual-Region)
        # ==========================================
        sim_a = t.matmul(batch_anchor_norm,   batch_anchor_norm.transpose(1,2))            # [B, win_size, win_size]
        sim_p = t.matmul(batch_positive_norm, batch_positive_norm.transpose(1,2))          # [B, win_size, win_size]
        sim_n = t.matmul(batch_negative_norm, batch_negative_norm.transpose(1,2))          # [B, win_size, win_size]
            
        dist_ap = (sim_a - sim_p)**2                                                        # [B, win_size, win_size]
        dist_an = (sim_a - sim_n)**2                                                        # [B, win_size, win_size]

        pos_sim_anomaly = t.sqrt((dist_ap * mask).sum(dim=(1,2)) / (anomaly_element + 1e-5) + 1e-5)        # [B]
        neg_sim_anomaly = t.sqrt((dist_an * mask).sum(dim=(1,2)) / (anomaly_element + 1e-5) + 1e-5)        # [B]

        pos_sim_normal = t.sqrt((dist_ap * (1-mask)).sum(dim=(1,2)) / (normal_element + 1e-5) + 1e-5)      # [B]
        neg_sim_normal = t.sqrt((dist_an * (1-mask)).sum(dim=(1,2)) / (normal_element + 1e-5) + 1e-5)      # [B]

        has_anom_matrix = (anomaly_element > 0).float()                                        # [B]
        has_norm_matrix = (normal_element > 0).float()                                         # [B]

        sim_loss_full = t.clamp(
            self.pos_ratio * (pos_sim_anomaly * has_anom_matrix + pos_sim_normal * has_norm_matrix) 
            + (1 - self.pos_ratio) * (neg_sim_normal * has_norm_matrix)                             
            - (neg_sim_anomaly * has_anom_matrix)                                                   
            + self.margin, 
            min=0
        )                                                                                        # [B]
        sim_loss = sim_loss_full * has_anom_matrix                                               # [B]
        
        # ==========================================
        # 2. MAGNITUDE / VARIANCE BRANCH
        # ==========================================
        has_anom_global = (hard_label.sum(dim=(1,2)) > 0).float()                                 # [B]
        
        if self.magnitude_mode.lower() in ["point_wise", "point-wise", "point", "p"]:
            mag_a = t.norm(batch_anchor, p=2, dim=-1)                                             
            mag_p = t.norm(batch_positive, p=2, dim=-1)                                        
            mag_n = t.norm(batch_negative, p=2, dim=-1)                                
    
            pos_mag_dist = ((mag_a - mag_p)**2) / ((mag_a + mag_p)**2 + 1e-5)          
            pos_mag_dist = pos_mag_dist.mean(dim=-1)                                   
    
            mag_label = hard_label if self.hard_mask == 1 else blur_label              
            mag_label = mag_label.squeeze(-1)                                          
            
            mag_anomaly_element = mag_label.sum(dim=-1)                                
            mag_normal_element = (1 - mag_label).sum(dim=-1)                           
    
            neg_mag_dist = ((mag_a - mag_n)**2) / ((mag_a + mag_n)**2 + 1e-5)          
            
            neg_mag_anomaly_dist = (neg_mag_dist * mag_label).sum(dim=-1) / (mag_anomaly_element + 1e-5)         
            neg_mag_normal_dist = (neg_mag_dist * (1 - mag_label)).sum(dim=-1) / (mag_normal_element + 1e-5)     
            
            mag_loss_full = t.clamp(self.pos_ratio*pos_mag_dist + (1-self.pos_ratio)*neg_mag_normal_dist \
                           - neg_mag_anomaly_dist + self.margin, min=0)            

            score_sim = (neg_sim_anomaly * has_anom_matrix - pos_sim_anomaly * has_anom_matrix) / (neg_sim_anomaly * has_anom_matrix + pos_sim_anomaly * has_anom_matrix + 1e-5)
            score_mag = (neg_mag_anomaly_dist - pos_mag_dist) / (neg_mag_anomaly_dist + pos_mag_dist + 1e-5)

        elif self.magnitude_mode.lower() in ["variance", "var", "v"]:
            anom_mask = hard_label if self.hard_mask == 1 else blur_label                           # [B, win_size, 1]
            norm_mask = 1.0 - anom_mask                                                             # [B, win_size, 1]
            
            std_a_anom = self.get_masked_std(batch_anchor, anom_mask)                                # [B, d_model]
            std_p_anom = self.get_masked_std(batch_positive, anom_mask)                              # [B, d_model]
            std_n_anom = self.get_masked_std(batch_negative, anom_mask)                              # [B, d_model]
    
            std_a_norm = self.get_masked_std(batch_anchor, norm_mask)                                # [B, d_model]
            std_p_norm = self.get_masked_std(batch_positive, norm_mask)                              # [B, d_model]
            std_n_norm = self.get_masked_std(batch_negative, norm_mask)                              # [B, d_model]

            def bounded_dist(v1, v2):                                                                # [B, d_model]
                return ((v1 - v2)**2) / ((v1 + v2)**2 + 1e-5)                                        # [B, d_model]

            pos_mag_anom_dist = bounded_dist(std_a_anom, std_p_anom).mean(dim=-1)                    # [B]
            pos_mag_norm_dist = bounded_dist(std_a_norm, std_p_norm).mean(dim=-1)                    # [B]
            
            has_norm_global = (norm_mask.sum(dim=(1,2)) > 0).float()                                # [B]
            has_anom_global = (hard_label.sum(dim=(1,2)) > 0).float()                               # [B]

            pos_mag_dist = (pos_mag_anom_dist * has_anom_global) + (pos_mag_norm_dist * has_norm_global)        # [B]
            neg_mag_anomaly_dist = bounded_dist(std_a_anom, std_n_anom).mean(dim=-1).view(-1)                   # [B]
            neg_mag_normal_dist = bounded_dist(std_a_norm, std_n_norm).mean(dim=-1).view(-1)                    # [B]

            var_loss_full = t.clamp(
                self.pos_ratio * pos_mag_dist 
                + (1 - self.pos_ratio) * (neg_mag_normal_dist * has_norm_global) 
                - (neg_mag_anomaly_dist * has_anom_global)                       
                + self.margin, 
                min=0
            )                                                 # [B]
            mag_loss_full = var_loss_full                     # [B]

        
            # Áp dụng t.clamp cho mẫu số để cấm nó rơi xuống mức quá nhỏ
            sim_denom = t.clamp(neg_sim_anomaly * has_anom_matrix + pos_sim_anomaly * has_anom_matrix, min=1e-5)
            score_sim = (neg_sim_anomaly * has_anom_matrix - pos_sim_anomaly * has_anom_matrix) / sim_denom

            mag_denom = t.clamp(neg_mag_anomaly_dist * has_anom_global + pos_mag_anom_dist * has_anom_global, min=1e-5)
            score_mag = (neg_mag_anomaly_dist * has_anom_global - pos_mag_anom_dist * has_anom_global) / mag_denom
        else:
            total_loss = self.reconstruct_weight * raw_recon_loss + \
                            self.contrastive_weight * sim_loss.mean(dim=0) 
            log_metrics = {
                "loss_recon": raw_recon_loss,
                "loss_contrastive": sim_loss.mean(dim=0),
                "loss_sim_raw": sim_loss.mean(dim=0),
                "loss_var_raw": 0,
                "recon_gap": 0, 
                "throttle": 0,
                "alpha_sim": 0,
                "alpha_mag": 0,
                "active_anom_ratio": 0,
                "latent_norm": 0
            }
            return total_loss, log_metrics

        mag_loss = mag_loss_full * has_anom_global            # [B]

        # ==========================================
        # 3. CONTRASTIVE LOSS (Adaptive Routing)
        # ==========================================
        score_sim, score_mag = score_sim.detach(), score_mag.detach()

        if B > 1:
            score_sim_std = score_sim.std(unbiased=False)
            score_mag_std = score_mag.std(unbiased=False)

            score_sim = (score_sim - score_sim.mean()) / (score_sim_std + 1e-5)                # [B]
            score_mag = (score_mag - score_mag.mean()) / (score_mag_std + 1e-5)                # [B]
        else:
            score_sim, score_mag = t.zeros_like(score_sim), t.zeros_like(score_mag)

        scores = t.stack([score_sim, score_mag], dim=-1)

        scaled_scores = scores / max(self.temperature, 1e-3)
        scaled_scores = scaled_scores - scaled_scores.max(dim=-1, keepdim=True)[0]
        
        alphas = t.softmax(scaled_scores, dim=-1)

        # ĐÃ SỬA LỖI SCALE: Bỏ nhân 2 để tổng alpha luôn = 1.0
        alpha_sim = self.alpha_floor + (1 - 2 * self.alpha_floor) * alphas[:, 0]                # [B]
        alpha_mag = self.alpha_floor + (1 - 2 * self.alpha_floor) * alphas[:, 1]                # [B]

        sample_contrastive_loss = (alpha_sim * sim_loss) + (alpha_mag * mag_loss)                # [B]
        raw_contrastive_loss = t.mean(sample_contrastive_loss)                                   # [1]

        # ==========================================
        # 4. GLOBAL MANIFOLD THROTTLING & REGULARIZATION
        # ==========================================
        recon_gap = (batch_recon_stat - batch_base_stat) / (batch_base_stat + 1e-5)                # [1]
        over_drift = t.relu(recon_gap - self.recon_tolerance)                                       # [1]
        throttle = t.exp(-self.throttle_beta * over_drift)
        throttled_contrastive_loss = raw_contrastive_loss * throttle                                # [1]

        # KHÓA CHẶT "ĐƯỜNG TẮT" BẰNG L2-NORM PENALTY 
        latent_norm_reg = t.sqrt(t.sum(batch_anchor**2, dim=-1) + 1e-5).mean()                     # [1]

        # ==========================================
        # FINAL LOSS
        # ==========================================
        recon_penalty = t.relu(raw_recon_loss - batch_base_mse)
        
        total_loss = self.reconstruct_weight * recon_penalty + self.contrastive_weight * throttled_contrastive_loss + 1e-4 * latent_norm_reg            
        log_metrics = {
            "loss_recon": raw_recon_loss.item(),
            "loss_contrastive": throttled_contrastive_loss.item(),
            "loss_sim_raw": sim_loss.mean().item(),
            "loss_var_raw": mag_loss.mean().item(),
            "recon_gap": recon_gap.mean().item(), 
            "throttle": throttle.item(),
            "alpha_sim": alpha_sim.mean().item(),
            "alpha_mag": alpha_mag.mean().item(),
            "active_anom_ratio": has_anom_global.mean().item(),
            "latent_norm": latent_norm_reg.item()
        }
        # for name, tensor in check_tensors.items():
        #     if t.isnan(tensor).any() or t.isinf(tensor).any():
        #         print(f"[NaN DETECTED] {name}")
        return total_loss, log_metrics
        
    def get_masked_std(self, x, m):
        # 1. Ép kiểu sang float32 để chặn đứng lỗi Tràn bộ nhớ (Overflow) của FP16
        x = x.float()
        m = m.float()
        
        valid_elements = m.sum(dim=1, keepdim=True) + 1e-5             
        local_mean = (x * m).sum(dim=1, keepdim=True) / valid_elements 
        
        # 2. Nhân mask TRƯỚC KHI bình phương. Nếu m=0 thì diff=0, không bao giờ có chuyện Inf * 0 = NaN
        diff = (x - local_mean) * m
        sum_sq = (diff ** 2).sum(dim=1)
        
        valid_elements_sq = valid_elements.squeeze(1)
        local_var = sum_sq / valid_elements_sq
        
        # 3. Kẹp giá trị chống số âm do sai số dấu phẩy động
        local_var = t.clamp(local_var, min=0.0)
        
        return t.sqrt(local_var + 1e-5)
