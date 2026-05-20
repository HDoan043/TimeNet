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

    def forward(self, x, x_hat, z, idx, pos_idx, neg_idx, labels, attn_pooling):
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
    def forward(self, x, x_hat, z, idx, pos_idx, neg_idx, labels, attn_pooling):
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
        self.cross_association = args.cross_association
        self.reconstruct_weight = args.reconstruct_weight
        self.contrastive_weight = args.contrastive_weight
        self.hard_mask = args.hard_mask
        self.max_ratio = args.max_ratio 
        self.pos_ratio = args.pos_ratio
        self.mse = nn.MSELoss()

        # Nhiệt độ cho Softmax
        self.temperature = getattr(args, 'temperature', 0.5)
        # Mức trọng số tối thiểu để cứu nhánh yếu (Chống sập)
        self.alpha_floor = getattr(args, 'alpha_floor', 0.2)
        
    def forward(self, x, x_hat, z, idx, pos_idx, neg_idx, labels, attn_pooling):
        '''
        My recommended loss: Semimi - Sequential Similarity (Fully Adaptive & Bounded)
        '''
        # same device
        idx = idx.to(x.device)
        pos_idx = pos_idx.to(x.device)
        neg_idx = neg_idx.to(x.device)
        attn_pooling = attn_pooling.to(x.device)
        labels = labels.to(x.device)

        # reconstruct loss (anchor only)
        B = idx.shape[0]
        recon_loss = self.mse(x[idx], x_hat[idx])
        
        batch_anchor = z[idx]                  # [B, win_size, d_model]            
        batch_positive = z[pos_idx]            # [B, win_size, d_model]            
        batch_negative = z[neg_idx]            # [B, win_size, d_model]   
        
        # normalization for similarity
        batch_anchor_norm = nn.functional.normalize(batch_anchor, p=2, dim=-1)
        batch_positive_norm = nn.functional.normalize(batch_positive, p=2, dim=-1)
        batch_negative_norm = nn.functional.normalize(batch_negative, p=2, dim=-1)    
            
        # attention on the abnormal timestamps base on labels
        blur_label = gaussian_blur_1d(labels).unsqueeze(-1)                        # [B, win_size, 1]
        hard_label = labels.unsqueeze(-1)                                          # [B, win_size, 1]
        hard_mask = self.max_ratio*t.maximum(hard_label, hard_label.transpose(1,2))\
             + (1-self.max_ratio)*t.matmul(hard_label, hard_label.transpose(1,2))  # [B, win_size, win_size]
        soft_mask = self.max_ratio*t.maximum(blur_label, blur_label.transpose(1,2))\
             + (1-self.max_ratio)*t.matmul(blur_label, blur_label.transpose(1,2))  # [B, win_size, win_size]
        mask = hard_mask if self.hard_mask == 1 else soft_mask
        
        anomaly_element = mask.sum(dim=(1,2))                                      # [B]
        normal_element = (1-mask).sum(dim=(1,2))                                   # [B]
            
        # ==========================================
        # 1. SIMILARITY BRANCH (Relation)
        # ==========================================
        sim_a = t.matmul(batch_anchor_norm,   batch_anchor_norm.transpose(1,2))    # [B, win_size, win_size]
        sim_p = t.matmul(batch_positive_norm, batch_positive_norm.transpose(1,2))  
        sim_n = t.matmul(batch_negative_norm, batch_negative_norm.transpose(1,2))  
            
        neg_anomaly_dist = (sim_a - sim_n)*mask                                  
        neg_anomaly_dist = (neg_anomaly_dist**2).sum(dim=(1,2))                  
        neg_anomaly_dist = t.sqrt(neg_anomaly_dist/(anomaly_element+1e-9) + 1e-9)  # [B]

        neg_normal_dist = (sim_a - sim_n)*(1-mask)                               
        neg_normal_dist = (neg_normal_dist**2).sum(dim=(1,2))                    
        neg_normal_dist = t.sqrt(neg_normal_dist/(normal_element+1e-9) + 1e-9)     # [B]
        
        pos_dist = t.sqrt(((sim_a - sim_p)**2).mean(dim=(1,2)) + 1e-9)             # [B]
        
        sim_loss = t.clamp(self.pos_ratio*pos_dist + (1-self.pos_ratio)*neg_normal_dist 
                       - neg_anomaly_dist + self.margin, min=0)                  # [B]

        # ==========================================
        # 2. MAGNITUDE BRANCH (Normalized Relative Distance)
        # ==========================================
        mag_a = t.norm(batch_anchor, p=2, dim=-1)                                  # [B, win_size]
        mag_p = t.norm(batch_positive, p=2, dim=-1)                                # [B, win_size]
        mag_n = t.norm(batch_negative, p=2, dim=-1)                                # [B, win_size]

        pos_mag_dist = ((mag_a - mag_p)**2) / ((mag_a + mag_p)**2 + 1e-9)          # [B, win_size]
        pos_mag_dist = pos_mag_dist.mean(dim=-1)                                   # [B]

        mag_label = hard_label if self.hard_mask == 1 else blur_label              
        mag_label = mag_label.squeeze(-1)                                          
        
        mag_anomaly_element = mag_label.sum(dim=-1)                                
        mag_normal_element = (1 - mag_label).sum(dim=-1)                           

        neg_mag_dist = ((mag_a - mag_n)**2) / ((mag_a + mag_n)**2 + 1e-9)          # [B, win_size]
        
        neg_mag_anomaly_dist = (neg_mag_dist * mag_label).sum(dim=-1) / (mag_anomaly_element + 1e-9)         # [B]
        neg_mag_normal_dist = (neg_mag_dist * (1 - mag_label)).sum(dim=-1) / (mag_normal_element + 1e-9)     # [B]
        
        mag_loss = t.clamp(self.pos_ratio*pos_mag_dist + (1-self.pos_ratio)*neg_mag_normal_dist 
                       - neg_mag_anomaly_dist + self.margin, min=0)                # [B]

        # ==========================================
        # 3. TOTAL LOSS (Bounded Adaptive Routing)
        # ==========================================
        contrastive_weight = self.contrastive_weight
        reconstruct_weight = self.args.reconstruct_weight

        # CẢI TIẾN 1: Dùng Relative Confidence Score [-1, 1] để đo Semantic (Đã Detach)
        score_sim = (neg_anomaly_dist - pos_dist) / (neg_anomaly_dist + pos_dist + 1e-9)
        score_sim = score_sim.detach()
        
        score_mag = (neg_mag_anomaly_dist - pos_mag_dist) / (neg_mag_anomaly_dist + pos_mag_dist + 1e-9)
        score_mag = score_mag.detach()

        # CẢI TIẾN 2: CHUẨN HÓA (TRƯỚC KHI SOFTMAX)
        if B > 1:
            score_sim = (score_sim - score_sim.mean()) / (score_sim.std() + 1e-9)
            score_mag = (score_mag - score_mag.mean()) / (score_mag.std() + 1e-9)
        else:
            score_sim = t.zeros_like(score_sim)
            score_mag = t.zeros_like(score_mag)

        # CẢI TIẾN 3: Đưa vào Softmax
        scores = t.stack([score_sim, score_mag], dim=-1)                           # [B,2]
        alphas = t.softmax(scores / self.temperature, dim=-1)                      # [B,2]

        # CẢI TIẾN 4: Dùng Alpha Floor chống sập (Dao động trong khoảng [floor, 1 - floor])
        # Ví dụ floor=0.2 -> alphas sẽ chạy từ 0.2 đến 0.8. Tổng 2 nhánh vẫn = 1.0
        alpha_sim_base = self.alpha_floor + (1 - 2 * self.alpha_floor) * alphas[:, 0]
        alpha_mag_base = self.alpha_floor + (1 - 2 * self.alpha_floor) * alphas[:, 1]

        # Nhân 2.0 để bảo toàn tổng năng lượng Gradient của hàm Loss ban đầu
        alpha_sim = alpha_sim_base * 2.0                                           # [B]
        alpha_mag = alpha_mag_base * 2.0                                           # [B]

        # Tính Loss Cuối cùng
        sample_contrastive_loss = (alpha_sim * sim_loss) + (alpha_mag * mag_loss)

        return reconstruct_weight*recon_loss + contrastive_weight*t.mean(mag_loss)
        # return reconstruct_weight*recon_loss + contrastive_weight*t.mean(sample_contrastive_loss)
        # else:
        #     sim_a_p = t.matmul(batch_anchor, batch_positive.transpose(1,2))   # [B, win_size, win_size]
        #     sim_a_n = t.matmul(batch_anchor, batch_negative.transpose(1,2))   # [B, win_size, win_size]
            
        #     # normalization
        #     sim_a_p = nn.functional.normalize(sim_a_p, p=2, dim=-1) 
        #     sim_a_n = nn.functional.normalize(sim_a_n, p=2, dim=-1) 
            
        #     # attention on the abnormal timestamps base on labels
        #     sim_a_n = sim_a_n*(1+self.label_guided_weight*label)    # [B, win_size, win_size]
            
        #     # loss
        #     win_size = batch_anchor.shape[1]
        #     i_matrix = t.eye(win_size, device=sim_a_p.device).unsqueeze(0) # [B, win_size, win_size]
        #     soft = 0.1
        #     i_matrix = t.ones_like(i_matrix, device=sim_a_p.device)*soft + (1-soft)*i_matrix
        #     pos_dist = t.norm(sim_a_p-i_matrix, p='fro', dim=(1,2)) # the representaton of timestamp i of anchor a should be the same as one of positive sample p
        #                                                             # where i of anchor a should be moderately the same as other timestamps'representation of positive sample p
        #     neg_dist = t.norm(sim_a_n-i_matrix, p='fro', dim=(1,2))
            
        #     loss = t.clamp(pos_dist - neg_dist + self.margin, min=0)     # [B]
        
