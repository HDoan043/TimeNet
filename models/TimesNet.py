import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.fft
from layers.Embed import DataEmbedding
from layers.Conv_Blocks import Inception_Block_V1
import time


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
    def __init__(self, configs):
        super(TimesBlock, self).__init__()
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.win_size = configs.win_size
        self.task_name = configs.task_name.lower()
        self.sample_length = self.seq_len + self.pred_len if self.task_name == "long_term_forecasting" else self.win_size
        self.k = configs.top_k
        # parameter-efficient design
        self.conv = nn.Sequential(
            Inception_Block_V1(configs.d_model, configs.d_ff,
                               num_kernels=configs.num_kernels),
            nn.GELU(),
            Inception_Block_V1(configs.d_ff, configs.d_model,
                               num_kernels=configs.num_kernels)
        )

    def forward(self, x):
        B, T, N = x.size()
        total_time = time.time()
    
        fft_time = time.time()
        period_list, period_weight = FFT_for_Period(x, self.k)
        fft_time = time.time() - fft_time

        res = []
        transform_time = 0
        inception_time = 0
        reshape_back_time = 0
        for i in range(self.k):
            trans_time = time.time()
            period = period_list[i]
            # padding
            if self.sample_length % period != 0:
                length = ((self.sample_length // period) + 1) * period
                padding = torch.zeros([x.shape[0], (length - self.sample_length), x.shape[2]]).to(x.device)
                out = torch.cat([x, padding], dim=1)
            else:
                length = self.sample_length
                out = x
            # reshape
            out = out.reshape(B, length // period, period,N).permute(0, 3, 1, 2).contiguous()
            trans_time = time.time() - trans_time
            transform_time += trans_time
            # 2D conv: from 1d Variation to 2d Variation
            incep_time = time.time()
            out = self.conv(out)
            incep_time = time.time() - incep_time
            inception_time += incep_time
            # reshape back
            reshape_time = time.time()
            out = out.permute(0, 2, 3, 1).reshape(B, -1, N)
            res.append(out[:, :self.sample_length, :])
            
            reshape_time = time.time() - reshape_time
            reshape_back_time += reshape_time

        combine_time = time.time()
        res = torch.stack(res, dim=-1)
        # adaptive aggregation
        period_weight = F.softmax(period_weight, dim=1)
        period_weight = period_weight.unsqueeze(
            1).unsqueeze(1).repeat(1, T, N, 1)
        res = torch.sum(res * period_weight, -1)
        combine_time = time.time() - combine_time
        # residual connection
        res = res + x
        total_time = time.time() - total_time
        
        return res


class Model(nn.Module):
    """
    Paper link: https://openreview.net/pdf?id=ju_Uqw384Oq
    """

    def __init__(self, configs):
        super(Model, self).__init__()
        self.configs = configs
        self.task_name = configs.task_name
        self.seq_len = configs.seq_len
        self.label_len = configs.label_len
        self.win_size = configs.win_size
        self.task_name = configs.task_name.lower()
        self.pred_len = configs.pred_len
        self.model = nn.ModuleList([TimesBlock(configs)
                                    for _ in range(configs.e_layers)])
        self.enc_embedding = DataEmbedding(configs.enc_in, configs.d_model, configs.embed, configs.freq,
                                           configs.dropout, configs.encode_timestamps)
        self.layer = configs.e_layers
        self.layer_norm = nn.LayerNorm(configs.d_model)

        if self.task_name == 'long_term_forecast' or self.task_name == 'short_term_forecast':
            self.predict_linear = nn.Linear(
                self.seq_len, self.pred_len + self.seq_len)
            self.projection = nn.Linear(
                configs.d_model, configs.c_out, bias=True)
        if self.task_name == 'imputation' or self.task_name == 'anomaly_detection' or self.task_name == 'supervise_5g_network':
            self.projection = nn.Linear(
                configs.d_model, configs.c_out, bias=True)
        if self.task_name == 'classification':
            self.act = F.gelu
            self.dropout = nn.Dropout(configs.dropout)
            self.projection = nn.Linear(
                configs.d_model * configs.seq_len, configs.num_class)
        if configs.contrastive == 1:
            self.contrastive_project = nn.Sequential(
                nn.Linear(configs.d_model, configs.d_model), 
                nn.ReLU(),
                nn.Linear(configs.d_model, configs.contrastive_d_model)
            )
            self.collapse_attn = nn.Linear(configs.contrastive_d_model, 1)
            
    def forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        total_time = time.time()
        # Normalization from Non-stationary Transformer
        norm_time = time.time()
        means = x_enc.mean(1, keepdim=True).detach()
        x_enc = x_enc.sub(means)
        stdev = torch.sqrt(
            torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
        x_enc = x_enc.div(stdev)

        norm_time = time.time() - norm_time
        # embedding
        embed_time = time.time()
        enc_out = self.enc_embedding(x_enc, x_mark_enc)  # [B,T,C]
        enc_out = self.predict_linear(enc_out.permute(0, 2, 1)).permute(
            0, 2, 1)  # align temporal dimension
    
        embed_time = time.time() - embed_time
        # TimesNet
        timesblock_time = time.time()
        for i in range(self.layer):
            enc_out = self.layer_norm(self.model[i](enc_out))
        timesblock_time = time.time() - timesblock_time
        # project back
        project_back_time = time.time()
        dec_out = self.projection(enc_out)
        project_back_time = time.time() - project_back_time

        # De-Normalization from Non-stationary Transformer
        denorm_time = time.time()
        dec_out = dec_out.mul(
                  (stdev[:, 0, :].unsqueeze(1).repeat(
                      1, self.pred_len + self.seq_len, 1)))
        dec_out = dec_out.add(
                  (means[:, 0, :].unsqueeze(1).repeat(
                      1, self.pred_len + self.seq_len, 1)))
        denorm_time = time.time() - denorm_time
    
        return dec_out

    def imputation(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask):
        # Normalization from Non-stationary Transformer
        means = torch.sum(x_enc, dim=1) / torch.sum(mask == 1, dim=1)
        means = means.unsqueeze(1).detach()
        x_enc = x_enc.sub(means)
        x_enc = x_enc.masked_fill(mask == 0, 0)
        stdev = torch.sqrt(torch.sum(x_enc * x_enc, dim=1) /
                           torch.sum(mask == 1, dim=1) + 1e-5)
        stdev = stdev.unsqueeze(1).detach()
        x_enc = x_enc.div(stdev)

        # embedding
        enc_out = self.enc_embedding(x_enc, x_mark_enc)  # [B,T,C]
        # TimesNet
        for i in range(self.layer):
            enc_out = self.layer_norm(self.model[i](enc_out))
        # project back
        dec_out = self.projection(enc_out)

        # De-Normalization from Non-stationary Transformer
        dec_out = dec_out.mul(
                  (stdev[:, 0, :].unsqueeze(1).repeat(
                      1, self.pred_len + self.seq_len, 1)))
        dec_out = dec_out.add(
                  (means[:, 0, :].unsqueeze(1).repeat(
                      1, self.pred_len + self.seq_len, 1)))
        return dec_out

    def supervise_5g_network(self, x_enc, x_enc_mark=None):
        # Normalization from Non-stationary Transformer
        means = x_enc.mean(1, keepdim=True).detach()
        x_enc = x_enc.sub(means)
        stdev = torch.sqrt(
            torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
        x_enc = x_enc.div(stdev)

        # embedding
        enc_out = self.enc_embedding(x_enc, x_enc_mark)  # [B,T,d_model]
        if self.configs.hidden_state_position.lower() == "base_model.enc_embedding":
            z = enc_out.clone()                              # [B,T,d_model]
        # TimesNet
        for i in range(self.layer):
            enc_out = self.layer_norm(self.model[i](enc_out))    # [B,T,d_model]
            if self.configs.hidden_state_position.lower() == f"base_model.{i}":
                z = enc_out.clone()                              # [B,T,d_model]

        # project back
        dec_out = self.projection(enc_out)

        # De-Normalization from Non-stationary Transformer
        sample_length = self.pred_len + self.seq_len if self.task_name == "long_term_forecasting" else self.win_size
        dec_out = dec_out.mul(
                  (stdev[:, 0, :].unsqueeze(1).repeat(
                      1, sample_length, 1)))
        dec_out = dec_out.add(
                  (means[:, 0, :].unsqueeze(1).repeat(
                      1, sample_length, 1)))
        
        return z, dec_out


    def anomaly_detection(self, x_enc, x_enc_mark=None):
        # Normalization from Non-stationary Transformer
        means = x_enc.mean(1, keepdim=True).detach()
        x_enc = x_enc.sub(means)
        stdev = torch.sqrt(
            torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
        x_enc = x_enc.div(stdev)

        # embedding
        enc_out = self.enc_embedding(x_enc, x_enc_mark)  # [B,T,d_model]
        if self.configs.contrastive == 1 and self.configs.hidden_state_position.lower() == "enc_embedding":
            z = enc_out.clone()                              # [B,T,d_model]
        # TimesNet
        for i in range(self.layer):
            enc_out = self.layer_norm(self.model[i](enc_out))    # [B,T,d_model]
            if self.configs.contrastive == 1 and self.configs.hidden_state_position.lower() == f"model.{i}":
                z = enc_out.clone()                              # [B,T,d_model]

        # project back
        dec_out = self.projection(enc_out)

        # De-Normalization from Non-stationary Transformer
        sample_length = self.pred_len + self.seq_len if self.task_name == "long_term_forecasting" else self.win_size
        dec_out = dec_out.mul(
                  (stdev[:, 0, :].unsqueeze(1).repeat(
                      1, sample_length, 1)))
        dec_out = dec_out.add(
                  (means[:, 0, :].unsqueeze(1).repeat(
                      1, sample_length, 1)))
        
        if self.configs.contrastive == 1:
            z = self.contrastive_project(z)                                    # z: [B, win_size, contrastive_d_model]
            attn = self.collapse_attn(z)                                       # attn: [3B, win_size, 1]
            attn_score = torch.softmax(attn, dim=1)                            # attn_score: [3B, win_size, 1]
            return z, dec_out, attn_score
        return dec_out

    def classification(self, x_enc, x_mark_enc):
        # embedding
        enc_out = self.enc_embedding(x_enc, None)  # [B,T,C]
        # TimesNet
        for i in range(self.layer):
            enc_out = self.layer_norm(self.model[i](enc_out))

        # Output
        # the output transformer encoder/decoder embeddings don't include non-linearity
        output = self.act(enc_out)
        output = self.dropout(output)
        # zero-out padding embeddings
        output = output * x_mark_enc.unsqueeze(-1)
        # (batch_size, seq_length * d_model)
        output = output.reshape(output.shape[0], -1)
        output = self.projection(output)  # (batch_size, num_classes)
        return output

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        if self.task_name == 'long_term_forecast' or self.task_name == 'short_term_forecast':
            dec_out = self.forecast(x_enc, x_mark_enc, x_dec, x_mark_dec)
            return dec_out[:, -self.pred_len:, :]  # [B, L, D]
        if self.task_name == 'imputation':
            dec_out = self.imputation(
                x_enc, x_mark_enc, x_dec, x_mark_dec, mask)
            return dec_out  # [B, L, D]
        if self.task_name == 'anomaly_detection':
            if self.configs.contrastive == 1:
                z, dec_out, attn_score = self.anomaly_detection(x_enc, x_mark_enc)
                return z, dec_out, attn_score
            else:
                dec_out = self.anomaly_detection(x_enc, x_mark_enc)
                return dec_out  # [B, L, D]
        if self.task_name == 'classification':
            dec_out = self.classification(x_enc, x_mark_enc)
            return dec_out  # [B, N]
        if self.task_name == 'supervise_5g_network':
            z, dec_out = self.supervise_5g_network(x_enc, x_mark_enc)
            return z, dec_out
        return None
