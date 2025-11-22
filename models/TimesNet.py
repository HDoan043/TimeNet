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
        ###############################################
        # print("***TimesBlock")
        # print("---- FFT ----")
        fft_time = time.time()
        ###############################################
        period_list, period_weight = FFT_for_Period(x, self.k)
        ###############################################
        fft_time = time.time() - fft_time
        # print("    ~ FFT time: {}s".format(fft_time))
        # print()
        # print("---- Loop for each period ----")
        ###############################################
        res = []
        transform_time = 0
        inception_time = 0
        reshape_back_time = 0
        for i in range(self.k):
            trans_time = time.time()
            period = period_list[i]
            # padding
            if (self.seq_len + self.pred_len) % period != 0:
                length = (((self.seq_len + self.pred_len) // period) + 1) * period
                padding = torch.zeros([x.shape[0], (length - (self.seq_len + self.pred_len)), x.shape[2]]).to(x.device)
                out = torch.cat([x, padding], dim=1)
            else:
                length = (self.seq_len + self.pred_len)
                out = x
            # reshape
            out = out.reshape(B, length // period, period,N).permute(0, 3, 1, 2).contiguous()
            #############################################
            trans_time = time.time() - trans_time
            transform_time += trans_time
            # print("    ~ Transform 1D to 2D: {}s".format(transform_time))
            #############################################
            # 2D conv: from 1d Variation to 2d Variation
            incep_time = time.time()
            out = self.conv(out)
            #############################################
            incep_time = time.time() - incep_time
            inception_time += incep_time
            # print("    ~ Inception block time: {}s".format(inception_time))
            #############################################
            # reshape back
            reshape_time = time.time()
            out = out.permute(0, 2, 3, 1).reshape(B, -1, N)
            res.append(out[:, :(self.seq_len + self.pred_len), :])
            #############################################
            reshape_time = time.time() - reshape_time
            reshape_back_time += reshape_time
            # print("    ~ Reshape back time: {}s".format(reshape_back_time))
            #############################################
        #############################################
        # print("----Combine----")
        combine_time = time.time()
        #############################################
        res = torch.stack(res, dim=-1)
        # adaptive aggregation
        period_weight = F.softmax(period_weight, dim=1)
        period_weight = period_weight.unsqueeze(
            1).unsqueeze(1).repeat(1, T, N, 1)
        res = torch.sum(res * period_weight, -1)
        #############################################
        combine_time = time.time() - combine_time
        # print("    ~ Combine time: {}s".format(combine_time))
        #############################################
        # residual connection
        res = res + x
        total_time = time.time() - total_time
        print("="*50)
        print("_ FFT: {}%".format(round(fft_time*100/total_time)))
        print("_ Transform: {}%".format(round(transform_time*100/total_time)))
        print("_ Inception: {}%".format(round(inception_time*100/total_time)))
        print("_ Reshape back: {}%".format(round(reshape_back_time*100/total_time)))
        print("_ Combine: {}%".format(round(combine_time*100/total_time)))
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
        self.pred_len = configs.pred_len
        self.model = nn.ModuleList([TimesBlock(configs)
                                    for _ in range(configs.e_layers)])
        self.enc_embedding = DataEmbedding(configs.enc_in, configs.d_model, configs.embed, configs.freq,
                                           configs.dropout)
        self.layer = configs.e_layers
        self.layer_norm = nn.LayerNorm(configs.d_model)
        if self.task_name == 'long_term_forecast' or self.task_name == 'short_term_forecast':
            self.predict_linear = nn.Linear(
                self.seq_len, self.pred_len + self.seq_len)
            self.projection = nn.Linear(
                configs.d_model, configs.c_out, bias=True)
        if self.task_name == 'imputation' or self.task_name == 'anomaly_detection':
            self.projection = nn.Linear(
                configs.d_model, configs.c_out, bias=True)
        if self.task_name == 'classification':
            self.act = F.gelu
            self.dropout = nn.Dropout(configs.dropout)
            self.projection = nn.Linear(
                configs.d_model * configs.seq_len, configs.num_class)

    def forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        total_time = time.time()
        # Normalization from Non-stationary Transformer
        #############################################
        # print("="*25 + "[1]-NORMALIZATION"+"="*25)
        norm_time = time.time()
        #############################################
        means = x_enc.mean(1, keepdim=True).detach()
        x_enc = x_enc.sub(means)
        stdev = torch.sqrt(
            torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
        x_enc = x_enc.div(stdev)

        ##############################################
        norm_time = time.time() - norm_time
        # print("- Normalization time: {}s".format(norm_time))
        # print()
        ##############################################
        # embedding
        ##############################################
        # print("="*25 + "[2]-EMBEDDING" + "="*25)
        embed_time = time.time()
        ##############################################
        enc_out = self.enc_embedding(x_enc, x_mark_enc)  # [B,T,C]
        enc_out = self.predict_linear(enc_out.permute(0, 2, 1)).permute(
            0, 2, 1)  # align temporal dimension
        ##############################################
        embed_time = time.time() - embed_time
        # print("- Embedding time: {}s".format(embed_time))
        ##############################################
        # TimesNet
        ##############################################
        # print()
        # print("="*25 + "[3]-TIMESBLOCK LAYERS" + "="*25)
        timesblock_time = time.time()
        ##############################################
        for i in range(self.layer):
            enc_out = self.layer_norm(self.model[i](enc_out))
        timesblock_time = time.time() - timesblock_time
        # project back
        ##############################################
        # print()
        # print("="*25 + "[4]-PROJECTION BACK" +"="*25)
        project_back_time = time.time()
        ##############################################
        dec_out = self.projection(enc_out)
        #############################################
        project_back_time = time.time() - project_back_time
        # print("- Project back time: {}s".format(project_back_time))
        #############################################

        # De-Normalization from Non-stationary Transformer
        #############################################
        # print()
        # print("="*25 + "[5]-DE NORMALIZATION" + "="*25)
        denorm_time = time.time()
        #############################################
        dec_out = dec_out.mul(
                  (stdev[:, 0, :].unsqueeze(1).repeat(
                      1, self.pred_len + self.seq_len, 1)))
        dec_out = dec_out.add(
                  (means[:, 0, :].unsqueeze(1).repeat(
                      1, self.pred_len + self.seq_len, 1)))
        #############################################
        denorm_time = time.time() - denorm_time
        # print("- De normalization time: {}s".format(denorm_time))
        #############################################
        total_time = time.time() - total_time
        # print("="*50)
        # print("_ Normalization: {}%".format(round(norm_time*100/total_time)))
        # print("_ Embedding: {}%".format(round(embed_time*100/total_time)))
        # print("_ TimesBlocks: {}%".format(round(timesblock_time*100/total_time)))
        # print("_ Project Back: {}%".format(round(project_back_time*100/total_time)))
        # print("_ De normalization: {}%".format(round(denorm_time*100/total_time)))
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

    def anomaly_detection(self, x_enc):
        # Normalization from Non-stationary Transformer
        means = x_enc.mean(1, keepdim=True).detach()
        x_enc = x_enc.sub(means)
        stdev = torch.sqrt(
            torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
        x_enc = x_enc.div(stdev)

        # embedding
        enc_out = self.enc_embedding(x_enc, None)  # [B,T,C]
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
            dec_out = self.anomaly_detection(x_enc)
            return dec_out  # [B, L, D]
        if self.task_name == 'classification':
            dec_out = self.classification(x_enc, x_mark_enc)
            return dec_out  # [B, N]
        return None
