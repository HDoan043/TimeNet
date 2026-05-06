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
    def __init__(self, temperature=0.1):
        super().__init__()
        self.temperature = temperature
        self.mse = nn.MSELoss()

    def forward(self, x, x_hat, z, idx, pos_idx, neg_idx, labels, neighbor_sim_anchor=1, neighbor_sim_pos=0, alpha=1.5):
        # reconstruct loss (anchor only)
        B = idx.shape[0]
        recon_loss = self.mse(x[idx], x_hat[idx])

        # normalize
        z = z.mean(dim=1)                                                    # z: [3*batch_size, 1, d_model]
        z = F.normalize(z, dim=1)                                            # z: [3*batch_size, 1, d_model]

        # similarity matrix (3B x 3B)
        sim = torch.matmul(z, z.T) / self.temperature                        # sim: [3*batch_size, 3*batch_size]

        # mask self similarity ( all similarity between the representation of a sample and itself are ignored)
        mask = torch.eye(sim.shape[0], device=sim.device).bool()
        sim.masked_fill_(mask, -torch.inf)

        # positive similarity
        # positive samples of an anchor are the windows near the anchor (distance from the anchor is small enough) and their augmentations
        pos_anchor_mask = torch.zeros((B,B), device=sim.device)                      # pos_anchor_mask: [B, B]
        for i in range(1, neighbor_sim_anchor+1):
            pos_anchor_mask.diagonal(offset=i).fill_(1)
            pos_anchor_mask.diagonal(offset=-i).fill_(1)
        pos_mask = torch.eye(B, device=sim.device)                                   # pos_mask: [B, B]
        for i in range(1, neighbor_sim_pos+1):
            pos_mask.diagonal(offset=i).fill_(1)
            pos_mask.diagonal(offset=-i).fill_(1)
        neg_mask = torch.zeros((B,B), device=sim.device)                             # neg_mask: [B, B]
        full_pos_mask = torch.cat([pos_anchor_mask, pos_mask, neg_mask], dim=1)      # full_pos_mask: [B, 3B], full_pos_mask[i,j] = 1 if sample[j] is a positive sample of anchor[i], = 0 else
        exp_sim = torch.exp(sim)
        pos_exp = exp_sim[idx] * full_pos_mask
        pos_count = full_pos_mask.sum(dim=1).clamp(min=1)                            # pos_count: [B]
        pos_sum = pos_exp.sum(dim=1)/pos_count                                       # pos_sum: [B]    
        
        # denominator
        full_neg_mask = ~full_pos_mask
        full_neg_mask.diagonal(offset=0).fill_(0)
        neg_sim = sim[idx].clone()                                                    # neg_sim: [B, 3B]
        neg_sim[~full_neg_mask] = -torch.inf
        exp_neg = torch.exp(neg_sim)
        weights = torch.ones_like(exp_neg, device=exp_neg.device)
        weights[idx][:,neg_idx] += alpha
        weights = (weights*full_neg_mask)
        # chọn top-k hardest negatives
        # k = int(0.1 * weights.shape[1])
        # hard_neg_mask = torch.zeros_like(weights)
        # topk_idx = torch.topk(neg_sim, k=k, dim=1).indices
        
        # hard_neg_mask.scatter_(1, topk_idx, 1)
        
        # weights = weights * (1 + alpha * hard_neg_mask)
        # weights = weights / weights.sum(dim=1, keepdim=True)
        
        denom = (exp_neg*weights).sum(dim=1)                        

        loss = -torch.log(pos_sum / denom)

        return recon_loss + loss.mean()
