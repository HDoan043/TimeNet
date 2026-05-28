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

    def reconstruct(self, x, x_hat, idx, pos_idx, neg_idx, hard_label):
        batch_anchor_win = x[idx]                         # [B, win_size, channel]
        batch_pos_win = x[pos_idx]                        # [B, win_size, channel]
        batch_neg_win = x[neg_idx]                        # [B, win_size, channel]
        batch_anchor_recon = x_hat[idx]                   # [B, win_size, channel]
        batch_pos_recon = x_hat[pos_idx]                  # [B, win_size, channel]
        batch_neg_recon = x_hat[neg_idx]                  # [B, win_size, channel]
        
        if self.reconstruct_negative:
            # reconstruct
            recon_anchor = self.mse_none(batch_anchor_win, batch_anchor_recon).mean(dim=(1,2))# [B]
            recon_pos = self.mse_none(batch_pos_win, batch_pos_recon).mean(dim=(1,2))         # [B]
            recon_neg = self.mse_none(batch_neg_win, batch_neg_recon).mean(dim=2)             # [B, win_size]
            
            anom_recon_neg = recon_neg*hard_label.squeeze(-1)                                 # [B, win_size]
            anom_elements = hard_label.sum(dim=1)                                             # [B]
            anom_recon_neg = anom_recon_neg.sum(dim=1)/(anom_elements + 1e-5)                 # [B]

            nor_recon_neg = recon_neg*(1-hard_label).squeeze(-1)                              # [B, win_size]
            nor_elements = (1-hard_label).squeeze(-1).sum(dim=1)                              # [B]
            nor_recon_neg = nor_recon_neg.sum(dim=1)/(nor_elements + 1e-5)                    # [B]

            pos_recon = t.stack([recon_anchor, recon_pos, nor_recon_neg], dim=1)              # [B, 3]
            
            # Lấy [0] để chọn values, bỏ qua indices
            max_pos_recon = t.max(pos_recon, dim=1)[0]                                        # [B]
            raw_recon_loss = t.relu(max_pos_recon - anom_recon_neg + self.margin).mean()      # [1]
        else: 
            # VÁ LỖI CÚ PHÁP: Dùng dim=(1,2) hoặc .mean()
            raw_recon_loss = self.mse_none(batch_anchor_win, batch_anchor_recon).mean()       # [1]
            
        return raw_recon_loss
        
    def forward(self, x, x_hat, z, idx, pos_idx, neg_idx, labels, attn_pooling, base_mse):
        # same device
        idx = idx.to(x.device)
        pos_idx = pos_idx.to(x.device)
        neg_idx = neg_idx.to(x.device)
        attn_pooling = attn_pooling.to(x.device)
        labels = labels.to(x.device)

        # reconstruct loss (anchor only)
        B = idx.shape[0]
        hard_label = labels.unsqueeze(-1)                                                  # [B, win_size, 1]
        recon_loss = self.reconstruct(x, x_hat, idx, pos_idx, neg_idx, hard_label)

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
        log_loss = {
            "loss_reconstruct": recon_loss.item(),
            "loss_contrastive": loss.item()
        }
        return reconstruct_weight*recon_loss + contrastive_weight*loss, log_loss

        
class TripletLoss(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.reconstruct_weight = args.reconstruct_weight
        self.contrastive_weight = args.contrastive_weight
        self.triplet = nn.TripletMarginLoss(margin=args.margin, p=2)
        
    def reconstruct(self, x, x_hat, idx, pos_idx, neg_idx, hard_label):
        batch_anchor_win = x[idx]                         # [B, win_size, channel]
        batch_pos_win = x[pos_idx]                        # [B, win_size, channel]
        batch_neg_win = x[neg_idx]                        # [B, win_size, channel]
        batch_anchor_recon = x_hat[idx]                   # [B, win_size, channel]
        batch_pos_recon = x_hat[pos_idx]                  # [B, win_size, channel]
        batch_neg_recon = x_hat[neg_idx]                  # [B, win_size, channel]
        
        if self.reconstruct_negative:
            # reconstruct
            recon_anchor = self.mse_none(batch_anchor_win, batch_anchor_recon).mean(dim=(1,2))# [B]
            recon_pos = self.mse_none(batch_pos_win, batch_pos_recon).mean(dim=(1,2))         # [B]
            recon_neg = self.mse_none(batch_neg_win, batch_neg_recon).mean(dim=2)             # [B, win_size]
            
            anom_recon_neg = recon_neg*hard_label.squeeze(-1)                                 # [B, win_size]
            anom_elements = hard_label.sum(dim=1)                                             # [B]
            anom_recon_neg = anom_recon_neg.sum(dim=1)/(anom_elements + 1e-5)                 # [B]

            nor_recon_neg = recon_neg*(1-hard_label).squeeze(-1)                              # [B, win_size]
            nor_elements = (1-hard_label).squeeze(-1).sum(dim=1)                              # [B]
            nor_recon_neg = nor_recon_neg.sum(dim=1)/(nor_elements + 1e-5)                    # [B]

            pos_recon = t.stack([recon_anchor, recon_pos, nor_recon_neg], dim=1)              # [B, 3]
            
            # Lấy [0] để chọn values, bỏ qua indices
            max_pos_recon = t.max(pos_recon, dim=1)[0]                                        # [B]
            raw_recon_loss = t.relu(max_pos_recon - anom_recon_neg + self.margin).mean()      # [1]
        else: 
            # VÁ LỖI CÚ PHÁP: Dùng dim=(1,2) hoặc .mean()
            raw_recon_loss = self.mse_none(batch_anchor_win, batch_anchor_recon).mean()       # [1]
            
        return raw_recon_loss
    def forward(self, x, x_hat, z, idx, pos_idx, neg_idx, labels, attn_pooling, base_mse):
        # same device
        idx = idx.to(x.device)
        pos_idx = pos_idx.to(x.device)
        neg_idx = neg_idx.to(x.device)
        attn_pooling = attn_pooling.to(x.device)
        labels = labels.to(x.device)
        
        # reconstruct loss (anchor only)
        B = idx.shape[0]
        hard_label = labels.unsqueeze(-1)                                                  # [B, win_size, 1]
        recon_loss = self.reconstruct(x, x_hat, idx, pos_idx, neg_idx, hard_label)

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
        log_loss = {
            "loss_reconstruct": recon_loss.item(),
            "loss_contrastive": loss.item()
        }
        return reconstruct_weight*recon_loss + contrastive_weight*loss, log_loss

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
        self.top_k_ratio = getattr(args, 'top_k_ratio', 0.3)
        
        self.mse_none = nn.MSELoss(reduction='none')

        self.temperature = getattr(args, 'temperature', 0.5)
        self.alpha_floor = getattr(args, 'alpha_floor', 0.2)
        
        self.recon_tolerance = getattr(args, 'recon_tolerance', 0.2) 
        self.throttle_beta = getattr(args, 'throttle_beta', 3.0) 

        self.magnitude_mode = getattr(args, 'magnitude_mode', 'variance')
        self.reconstruct_negative = getattr(args, 'reconstruct_negative', 0)

    def reconstruct(self, x, x_hat, idx, pos_idx, neg_idx, hard_label):
        batch_anchor_win = x[idx]                         # [B, win_size, channel]
        batch_pos_win = x[pos_idx]                        # [B, win_size, channel]
        batch_neg_win = x[neg_idx]                        # [B, win_size, channel]
        batch_anchor_recon = x_hat[idx]                   # [B, win_size, channel]
        batch_pos_recon = x_hat[pos_idx]                  # [B, win_size, channel]
        batch_neg_recon = x_hat[neg_idx]                  # [B, win_size, channel]
        
        if self.reconstruct_negative:
            # reconstruct
            recon_anchor = self.mse_none(batch_anchor_win, batch_anchor_recon).mean(dim=(1,2))# [B]
            recon_pos = self.mse_none(batch_pos_win, batch_pos_recon).mean(dim=(1,2))         # [B]
            recon_neg = self.mse_none(batch_neg_win, batch_neg_recon).mean(dim=2)             # [B, win_size]
            
            anom_recon_neg = recon_neg*hard_label.squeeze(-1)                                 # [B, win_size]
            anom_elements = hard_label.squeeze(-1).sum(dim=1)                                 # [B]
            # anom_recon_neg = anom_recon_neg.sum(dim=1)/(anom_elements + 1e-5)                 # [B]

            # Tìm các cửa sổ có chứa lỗi
            valid_windows = anom_elements > 0
            
            if valid_windows.any():
                anom_loss_total = 0.0
                valid_count = 0
                
                # (BUỘC PHẢI DÙNG VÒNG LẶP trên các cửa sổ hợp lệ vì mỗi cửa sổ có số K khác nhau)
                # Tuy nhiên vì số lượng cửa sổ lỗi trong batch nhỏ (vd 5-10 cái), vòng lặp này rất nhanh.
                for b in range(hard_label.shape[0]):
                    if valid_windows[b]:
                        # Tính K cho cửa sổ này
                        k = max(1, int(num_anom_per_window[b].item() * self.top_k_ratio))
                        
                        # Lấy Top K loss của cửa sổ b
                        topk_loss, _ = t.topk(anom_recon_neg[b], k)
                        
                        # Cộng dồn trung bình
                        anom_loss_total += topk_loss.mean()
                        valid_count += 1
                
                anom_loss = anom_loss_total / valid_count
            else:
                anom_loss = 0.0
            
            anom_recon_neg = anom_loss
            nor_recon_neg = recon_neg*(1-hard_label).squeeze(-1)                              # [B, win_size]
            nor_elements = (1-hard_label).squeeze(-1).sum(dim=1)                              # [B]
            nor_recon_neg = nor_recon_neg.sum(dim=1)/(nor_elements + 1e-5)                    # [B]

            pos_recon = t.stack([recon_anchor, recon_pos, nor_recon_neg], dim=1)              # [B, 3]
            
            # Lấy [0] để chọn values, bỏ qua indices
            max_pos_recon = t.max(pos_recon, dim=1)[0]                                        # [B]
            raw_recon_loss = t.relu(max_pos_recon - anom_recon_neg + self.margin).mean()      # [1]
        else: 
            # VÁ LỖI CÚ PHÁP: Dùng dim=(1,2) hoặc .mean()
            raw_recon_loss = self.mse_none(batch_anchor_win, batch_anchor_recon).mean()       # [1]
            
        return raw_recon_loss

    def similarity(self, batch_anchor_norm, batch_positive_norm, batch_negative_norm, mask, anomaly_element, normal_element):
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

        return sim_loss.mean()                                                                    #[1]

    def magnitude(self, hard_label, blur_label, batch_anchor, batch_positive, batch_negative):
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

            # score_sim = (neg_sim_anomaly * has_anom_matrix - pos_sim_anomaly * has_anom_matrix) / (neg_sim_anomaly * has_anom_matrix + pos_sim_anomaly * has_anom_matrix + 1e-5)
            # score_mag = (neg_mag_anomaly_dist - pos_mag_dist) / (neg_mag_anomaly_dist + pos_mag_dist + 1e-5)

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
            # sim_denom = t.clamp(neg_sim_anomaly * has_anom_matrix + pos_sim_anomaly * has_anom_matrix, min=1e-5)
            # score_sim = (neg_sim_anomaly * has_anom_matrix - pos_sim_anomaly * has_anom_matrix) / sim_denom

            # mag_denom = t.clamp(neg_mag_anomaly_dist * has_anom_global + pos_mag_anom_dist * has_anom_global, min=1e-5)
            # score_mag = (neg_mag_anomaly_dist * has_anom_global - pos_mag_anom_dist * has_anom_global) / mag_denom

            mag_loss_full = mag_loss_full * has_anom_global            # [B]
        return mag_loss_full.mean()

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
        # Reconstruct anchor window
        sample_recon = self.mse_none(x[idx], x_hat[idx]).mean(dim=(1,2))             # [B]
        raw_recon_loss = sample_recon.mean()                                         # [1]

        if B > 1:
            batch_base_stat = t.quantile(base_mse, 0.8)                                # [1]
            batch_recon_stat = t.quantile(sample_recon.detach(), 0.8)                  # [1]
        else:
            batch_base_stat = base_mse.squeeze()
            batch_recon_stat = sample_recon.detach().squeeze()

        # Reconstruct negative window
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

        mask_label = hard_label if self.hard_mask == 1 else blur_label                     # [B, win_size, 1]
        mask = self.max_ratio*t.maximum(mask_label, mask_label.transpose(1,2))\
             + (1-self.max_ratio)*t.matmul(mask_label, mask_label.transpose(1,2))          # [B, win_size, win_size]
        
        anomaly_element = mask.sum(dim=(1,2))                                              # [B]
        normal_element = (1-mask).sum(dim=(1,2))                                           # [B]

        recon_loss = self.reconstruct(x, x_hat, idx, pos_idx, neg_idx, hard_label)         # [1]
        sim_loss = self.similarity(batch_anchor_norm, batch_positive_norm, batch_negative_norm, mask, anomaly_element, normal_element) #[1]
        total_loss = self.reconstruct_weight*recon_loss + self.contrastive_weight*sim_loss

        log_metrics = {
            "loss_reconstruct": (recon_loss).item(),
            "loss_contrastive": (sim_loss).item(),
            # "loss_sim_raw": sim_loss.item(),
            # "loss_var_raw": 0,
            # "recon_gap": 0, 
            # "throttle": 1.0, # Giả lập throttle đang mở full
            # "alpha_sim": 1.0, # Đang dùng 100% sim
            # "alpha_mag": 0,
            # "active_anom_ratio": (hard_label.sum(dim=(1,2)) > 0).float().mean().item(),
            # "latent_norm": 0
        }
        
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
            
class CorrectorBCELoss(nn.Module):
    def __init__(self, args):
        super(CorrectorLoss, self).__init__()
        self.margin = args.margin
        self.bce = nn.BCELoss(reduction='none')
        # Dùng BCE (Binary Cross Entropy) để ép điểm về 0 hoặc 1 là chuẩn nhất
        
    def forward(self, final_score, labels):
        """
        final_score: [B, win_size] (Điểm do Bi-LSTM xuất ra, đã qua relu)
        labels: [B, win_size] (Nhãn thực tế từ file csv, 0 là bình thường/fake, 1 là lỗi)
        """
        # 1. Vì final_score có thể lớn hơn 1 (do Relu), ta cần dùng Sigmoid để kẹp nó về [0, 1]
        # Điều này giúp hàm BCE không bị văng lỗi "Input must be between 0 and 1"
        prob_score = t.sigmoid(final_score)
        
        # 2. Tính Binary Cross Entropy Loss
        # Nếu label=1, ép prob_score -> 1 (tức là final_score càng lớn càng tốt)
        # Nếu label=0, ép prob_score -> 0 (tức là final_score càng nhỏ càng tốt)
        loss_matrix = self.bce(prob_score, labels.float()) # [B, win_size]
        
        # 3. Tính Loss tổng
        total_loss = loss_matrix.mean()
        return total_loss

# Multiple Instance Leanring
class CorrectorMIL_Loss(nn.Module):
    def __init__(self, args):
        super(CorrectorMIL_Loss, self).__init__()
        self.args = args
        self.bce_none = nn.BCELoss(reduction='none')
        # top_k_ratio = 0.2 nghĩa là ta chỉ ép 20% số điểm cao nhất trong vùng lỗi phải tiến về 1.
        # 80% còn lại tha bổng.
        self.top_k_ratio = args.top_k_ratio 
        
    def forward(self, final_score, labels):
        # 1. Ép về [0, 1]
        prob_score = t.sigmoid(final_score)
        
        # 2. Tính BCE Loss cho TỪNG ĐIỂM
        point_loss = self.bce_none(prob_score, labels.float()) # [B, win_size]
        
        # 3. Phân rã Mask
        normal_mask = (labels == 0).float()
        anom_mask = (labels == 1).float()
        
        # 4. LOSS VÙNG BÌNH THƯỜNG (Ép TẤT CẢ xuống)
        normal_loss = (point_loss * normal_mask).sum() / (normal_mask.sum() + 1e-5)
        
        # 5. LOSS VÙNG BẤT THƯỜNG (Top-K trên TỪNG CỬA SỔ)
        # Chỉ giữ lại loss của các điểm nhãn 1, các điểm nhãn 0 bị ép về 0
        anom_loss_values = point_loss * anom_mask # [B, win_size]
        
        # Đếm số lượng điểm lỗi TRONG TỪNG CỬA SỔ
        num_anom_per_window = anom_mask.sum(dim=1) # [B]
        
        # Tìm các cửa sổ có chứa lỗi
        valid_windows = num_anom_per_window > 0
        
        if valid_windows.any():
            anom_loss_total = 0.0
            valid_count = 0
            
            # (BUỘC PHẢI DÙNG VÒNG LẶP trên các cửa sổ hợp lệ vì mỗi cửa sổ có số K khác nhau)
            # Tuy nhiên vì số lượng cửa sổ lỗi trong batch nhỏ (vd 5-10 cái), vòng lặp này rất nhanh.
            for b in range(labels.shape[0]):
                if valid_windows[b]:
                    # Tính K cho cửa sổ này
                    k = max(1, int(num_anom_per_window[b].item() * self.top_k_ratio))
                    
                    # Lấy Top K loss của cửa sổ b
                    topk_loss, _ = t.topk(anom_loss_values[b], k)
                    
                    # Cộng dồn trung bình
                    anom_loss_total += topk_loss.mean()
                    valid_count += 1
            
            anom_loss = anom_loss_total / valid_count
        else:
            anom_loss = 0.0
            
        # 6. Tổng Loss
        total_loss = normal_loss + anom_loss
        
        return total_loss
