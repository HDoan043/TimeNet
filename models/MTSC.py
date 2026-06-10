'''
TimesNet from "TimesNet: Temporal 2D-Variation Modeling for General Time Series Analysis" (ICLR 2023)
Code partially from https://github.com/thuml/Time-Series-Library/

Copyright (c) 2021 THUML @ Tsinghua University
'''

from typing import Dict
import numpy as np
import torchinfo
import torch
from torch import nn, optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torch.fft
from torch.nn.utils import weight_norm
import math
import tqdm
import os

from ..utils.torch_utility import EarlyStoppingTorch, DataEmbedding, adjust_learning_rate, get_gpu
from ..utils.dataset import ReconstructDataset    

class Inception_Block_V1(nn.Module):
    def __init__(self, in_channels, out_channels, num_kernels=6, init_weight=True):
        super(Inception_Block_V1, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_kernels = num_kernels
        kernels = []
        for i in range(self.num_kernels):
            kernels.append(nn.Conv2d(in_channels, out_channels, kernel_size=2 * i + 1, padding=i))
        self.kernels = nn.ModuleList(kernels)
        if init_weight:
            self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        res_list = []
        for i in range(self.num_kernels):
            res_list.append(self.kernels[i](x))
        res = torch.stack(res_list, dim=-1).mean(-1)
        return res


def FFT_for_Period(x, k=2):
    # [B, T, C]
    xf = torch.fft.rfft(x, dim=1)
    # find period by amplitudes
    frequency_list = abs(xf).mean(0).mean(-1)
    frequency_list[0] = 0
    _, top_list = torch.topk(frequency_list, k)
    top_list = top_list.detach().cpu().numpy()
    period = x.shape[1] // top_list
    return period, abs(xf).mean(-1)[:, top_list]


class TimesBlock(nn.Module):
    def __init__(self,
                 seq_len=96,
                 pred_len=0,
                 top_k=3,
                 d_model=8,
                 d_ff=16,
                 num_kernels=6,
                 expert = False
                 ):
        super(TimesBlock, self).__init__()
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.k = top_k
        # parameter-efficient design
        d_model = d_model if not expert else 1
        d_ff = d_ff if not expert else 4
        self.conv = nn.Sequential(
            Inception_Block_V1(d_model, d_ff,
                               num_kernels=num_kernels),
            nn.GELU(),
            Inception_Block_V1(d_ff, d_model,
                               num_kernels=num_kernels)
        )

    def forward(self, x):
        B, T, N = x.size()                                              # [B*c_in, win_size, 1]
        period_list, period_weight = FFT_for_Period(x, self.k)

        res = []
        for i in range(self.k):
            period = period_list[i]
            # padding
            if (self.seq_len + self.pred_len) % period != 0:
                length = (
                                 ((self.seq_len + self.pred_len) // period) + 1) * period
                padding = torch.zeros([x.shape[0], (length - (self.seq_len + self.pred_len)), x.shape[2]]).to(x.device)
                out = torch.cat([x, padding], dim=1)
            else:
                length = (self.seq_len + self.pred_len)
                out = x
            # reshape
            out = out.reshape(B, length // period, period,
                              N).permute(0, 3, 1, 2).contiguous()
            # 2D conv: from 1d Variation to 2d Variation
            out = self.conv(out)
            # reshape back
            out = out.permute(0, 2, 3, 1).reshape(B, -1, N)
            res.append(out[:, :(self.seq_len + self.pred_len), :])
        res = torch.stack(res, dim=-1)
        # adaptive aggregation
        period_weight = F.softmax(period_weight, dim=1)
        period_weight = period_weight.unsqueeze(
            1).unsqueeze(1).repeat(1, T, N, 1)
        res = torch.sum(res * period_weight, -1)
        # residual connection
        res = res + x
        return res                                                          # [B*c_in, win_size, 1]


class TimesNetExpert(nn.Module):
    """
    Paper link: https://openreview.net/pdf?id=ju_Uqw384Oq
    """

    def __init__(self,
                 seq_len=96,
                 pred_len=0,
                 d_model=8,
                 e_layers=1,
                 num_kernels=6,
                 top_k = 3,
                 dropout=0.1,
                 ):
        super(TimesNetExpert, self).__init__()
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.model = nn.ModuleList([
            TimesBlock(seq_len=self.seq_len, 
                       top_k=top_k, 
                       d_model=d_model, 
                       d_ff=4*d_model, 
                       num_kernels=num_kernels,
                       expert=True) for _ in range(e_layers)
            ])
        self.layer = e_layers
        self.project = nn.Linear(in_features=seq_len, out_features=d_model)

    def anomaly_detection(self, x_enc):
        enc_out = x_enc
        # TimesNet
        for i in range(self.layer):
            enc_out = self.model[i](enc_out)                           # [B*c_in, win_size, 1]
        
        dec_out = self.project(enc_out.squeeze(-1))                    # [B*c_in, d_model]
        return dec_out

    def forward(self, x_enc):
        dec_out = self.anomaly_detection(x_enc)
        return dec_out  # [B, L, D]

class Model(nn.Module):
    def __init__(self, configs):
        super(Model, self).__init__()
        num_experts = configs.num_experts
        c_in = configs.c_in
        d_model = configs.d_model
        win_size = configs.win_size
        expert_layers = configs.e_layers
        expert_top_k = configs.top_k
        self.c_in = c_in
        self.d_model = d_model
        self.num_experts = num_experts
        self.expert_represent_vectors = nn.Parameter(torch.randn(1, win_size, num_experts))
        
        self.experts_ls =  nn.ModuleList([
            TimesNetExpert(
                 seq_len=win_size,
                 pred_len=0,
                 d_model=d_model,
                 e_layers=expert_layers,
                 num_kernels = expert_num_kernels,
                 top_k = expert_top_k,
                 dropout=0.1) for _ in range(num_experts)])
        self.channel_attention = nn.MultiheadAttention(embed_dim=d_model, num_heads=4, batch_first=True, dropout=0.1)
        self.norm = nn.LayerNorm(d_model)
        
        self.decoder = nn.Linear(in_features= d_model, out_features=win_size)
        
    def forward(self, x, x_mark, x_dec, x_mark_dec):                                   # [B, win_size, c_in]
        # Normalization from Non-stationary Transformer
        means = x.mean(1, keepdim=True).detach()
        x = x - means
        stdev = torch.sqrt(
            torch.var(x, dim=1, keepdim=True, unbiased=False) + 1e-5)
        x /= stdev
        x = x.permute(0,2,1)                                # [B, c_in, win_size]
        
        x_norm = F.normalize(x, p=2, dim=2)                 # [B, c_in, win_size]
        
        exp_vec_norm = F.normalize(self.expert_represent_vectors, p=2, dim=1)
        expert_ratio = x_norm @ exp_vec_norm                # [B, c_in, num_experts]
        expert_ratio = torch.softmax(expert_ratio, dim=-1)  # [B, c_in, num_experts]
        expert_ratio = expert_ratio.unsqueeze(-1)           # [B, c_in, num_experts, 1]
        
        out1 = []
        B, C, win_size = x.shape
        x_flat = x.contiguous().view(B*C, win_size, 1)      # [B*c_in, win_size, 1]
        
        for expert in self.experts_ls:
            out1.append(expert(x_flat))                       # [B*c_in, d_model] x num_expert
            
        out1 = torch.stack(out1, dim=0)                       # [num_expert, B*c_in, d_model]
        out1 = out1.permute(1,0,2).contiguous()               # [B*c_in, num_expert, d_model]
        out1 = torch.reshape(out1, (B,C,self.num_experts, -1))# [B, c_in, num_expert, d_model]
        out1 = (out1*expert_ratio).sum(dim=2)                 # [B, c_in, d_model]
        
        attn_out, attn_weights = self.channel_attention(out1, out1, out1)
        
        out = self.norm(out1 + attn_out)
        
        x_hat = self.decoder(out)                             # [B, c_in, win_size]
        x_hat = x_hat.permute(0,2,1).contiguous()             # [B, win_size, c_in]
        
        # Denormalization
        x_rec = x_hat * stdev + means
        
        return x_rec
