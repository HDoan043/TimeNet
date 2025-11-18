import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.fft
from layers.Embed import DataEmbedding
from layers.Conv_Blocks import Inception_Block_V1


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


class TimesBlockUpdate(nn.Module):
    def __init__(self, configs):
        super(TimesBlockUpdate, self).__init__()
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
      
        self.att_inner = nn.MultiheadAttention(configs.d_model, configs.n_heads, batch_first = True)
        self.att_outer = nn.MultiheadAttention(configs.d_model, configs.n_heads, batch_first = True)
        self.att_combine = nn.MultiheadAttention(configs.d_model, configs.n_heads, batch_first = True)

        inner_mlp = []
        outer_mlp = []
        for _ in range(2):
            inner_mlp.extend([
                nn.Linear(configs.d_model, 1024),
                nn.GELU(),
                nn.Linear(1024, configs.d_model),
                nn.GELU()])
            outer_mlp.extend([
                nn.Linear(configs.d_model, 1024),
                nn.GELU(),
                nn.Linear(1024, configs.d_model),
                nn.GELU()])
        
        self.feedforward_inner = nn.Sequential(*inner_mlp)
        self.feedforward_outer = nn.Sequential(*outer_mlp)
        
        ff_mlp = []
        for _ in range(configs.d_ff):
            ff_mlp.extend( [
                nn.Linear(configs.d_model, 1024),
                nn.GELU(),
                nn.Linear(1024, configs.d_model),
                nn.GELU()])
        self.feedforward = nn.Sequential( *ff_mlp)

    def forward(self, x_inner, x_outer):                                     # x_inner: [Batch_size x period x f x d_model], 
                                                                             # x_outer: [Batch_size x period x f x d_model]
        B, P, F, D = x_inner.shape

        x_inner = torch.reshape( x_inner, (B*F, P, D))
        x_outer = torch.reshape( x_outer, (B*P, F, D))
        
        # inner
        p = x_inner.shape[1]
        att_in = self.att_inner(x_inner, x_inner, x_inner)
        x_inner = x_inner + att_in
        x_inner = self.feedforward_inner(x_inner)                            # x_in: [Batch_size*f x period x d_model]

        # outer
        f = x_outer.shape[2]
        att_out = self.att_outer(x_outer, x_outer, x_outer)
        x_outer = x_outer + att_out
        x_outer = self.feedforward_outer(x_outer)                            # x_out: [Batch_size *period x f x d_model]

        # combine inner and outer
        x_inner = torch.reshape( x_inner, (B*P*F, D))                        # x_inner: [Batch_size * period * f x d_model]
        x_outer = torch.reshape( x_outer, (B*P*F, D))                        # x_outer: [Batch_size * period * f x d_model]
        x = torch.cat([x_inner, x_outer], dim=0)                             # x: [2* Batch_size * period * f x d_model]
        att_combine = self.att_combine(x, x, x)
        x = x + atta_combine                                                 # x: [2* Batch_size * period * f x d_model]
        x = self.feedforward(x)                                              # x: [2 * Batch_size * period * f x d_model]

        length = x.shape[0]
        new_x_inner = x[:length/2,:]                                         # new_x_inner: [Batch_size * period * f x d_model]
        new_x_inner = torch.reshape(new_x_inner, (B, P, F, D))               # new_x_inner: [Batch_size x period x f x d_model]
        new_x_outer = x[length/2:, :]                                        # new_x_outer: [Batch_size * period * f x d_model]
        new_x_outer = torch.reshape(new_x_outer, (B, P, F, D))               # new_x_outer: [Batch_size x period x f x d_model]
        
        return new_x_inner, new_x_outer

class CombineHead(nn.Module):
    def __init__(self, d_model):
        super(CombineHead, self).__init__()
        self.d_model = d_model
        self.combine = nn.Linear( 2*d_model, d_model)
    def forward(self, x_p, x_f):                                            # x_p: [Batch_size x period x f x d_model]
                                                                            # x_f: [Batch_size x period x f x d_model]
        B, P, F, D = x_p.shape
        x_p = torch.reshape(x_p, (B*P*F, D))                                # x_p: [Batch_size * period * f x d_model]
        x_f = torch.reshape(x_f, (B*P*F, D))                                # x_f: [Batch_size * period * f x d_model]
        x = torch.cat([x_p, x_f], dim=1)                                    # x: [Batch_size * period * f x 2 * d_model]
        x = self.combine(x)                                                 # x: [Batch_size * period *f x d_model]
        x = torch.reshape( x, (B, P, F, D))                                 # x: [Batch_size x period x f x d_model]

        return x
        
class Model(nn.Module):
    """
    Paper link: https://openreview.net/pdf?id=ju_Uqw384Oq
    """

    def __init__(self, configs):
        super(Model, self).__init__()
        self.configs = configs
        self.periods = {} # dictionary saving the period corresponding to each sample index, this is for avoiding running FFT to calculate periods twice
        self.task_name = configs.task_name
        self.seq_len = configs.seq_len
        self.label_len = configs.label_len
        self.pred_len = configs.pred_len
        self.model = nn.ModuleList([TimesBlockUpdate(configs)
                                    for _ in range(configs.e_layers)])
        self.enc_embedding = DataEmbedding(configs.enc_in, configs.d_model, configs.embed, configs.freq,
                                           configs.dropout)
        self.layer = configs.e_layers
        self.layer_norm = nn.LayerNorm(configs.d_model)
        
        if self.task_name == 'long_term_forecast' or self.task_name == 'short_term_forecast':
            self.predict_linear = nn.Linear(
                self.seq_len, self.pred_len + self.seq_len)
            self.combine_head = CombineHead(configs.d_model)
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

    def forecast(self, index_x, x_enc, x_mark_enc, x_dec, x_mark_dec): 
        # If index of sample x is not in dictionary self.periods -> x has not been calculated FFT yet
        if index_x not in self.periods:
            period, _ = FFT_for_Period(x_enc, k=1)
            self.periods[index_x] = period
  
        # Normalization from Non-stationary Transformer
        means = x_enc.mean(1, keepdim=True).detach()
        x_enc = x_enc.sub(means)
        stdev = torch.sqrt(
            torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
        x_enc = x_enc.div(stdev)

        # TRANSFORM 1D -> 2D
        # padding
        B = x_enc.shape[0]
        period = self.periods[index_x]
        if (self.seq_len + self.pred_len) % period != 0:
            length = (((self.seq_len + self.pred_len) // period) + 1) * period
            padding = torch.zeros([x_enc.shape[0], (length - (self.seq_len + self.pred_len)), x_enc.shape[2]]).to(x_enc.device)
            out = torch.cat([x_enc, padding], dim=1)
        else:
            length = (self.seq_len + self.pred_len)
            out = x_enc
        # reshape
        out = out.reshape(B, length // period, period,
                          N).contiguous()    # out: [Batch_size x period_i x f_i x nvars]
      
        # EMBEDDING
        B, P, F, N = out.shape
        # [1] - Embedding with dimension of f_i: Embedding for inside a period
        out_f = torch.reshape( out, (B*F, P, N) )              # out_f: [Batch_size*f_i x period_i x nvars]
        out_f = self.enc_embedding(out_f)                      # out_f: [Batch_size*f_i x period_i x d_model]
        out_f = torch.reshape( out_f, (B, P, F, -1))           # out_f: [Batch_size x period_i x f_i x d_model]
    
        # [2] - Embedding with dimension of p_i: Embedding across the period
        out_p = torch.reshape( out, (B*P, F, N))               # out_p: [Batch_size*period_i x f_i x nvars]
        out_p = self.enc_embedding(out_p)                      # out_p: [Batch_size*period_i x f_i x d_model]
        out_p = torch.reshape( out_f, (B, P, F, -1))           # out_p: [Batch_size x period_i x f_i x d_model]

        # PASS IN BACKBONE
        for each in self.model:
            out_f, out_p = each(out_f, out_p)
            out_f = self.layer_norm(out_f)
            out_p = self.layer_norm(out_p)

        # Combine out_f and out_p
        enc_out = self.combine_head(out_f, out_p)              # enc_out: [Batch_size x period_i x f_i x d_model]
        # project back
        dec_out = self.projection(enc_out)                     # dec_out: [Batch_size x period_i x f_i x nvars]

        # De-Normalization from Non-stationary Transformer
        dec_out = dec_out.mul(
                  (stdev[:, 0, :].unsqueeze(1).repeat(
                      1, self.pred_len + self.seq_len, 1)))
        dec_out = dec_out.add(
                  (means[:, 0, :].unsqueeze(1).repeat(
                      1, self.pred_len + self.seq_len, 1)))
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

    def forward(self, index, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        if self.task_name == 'long_term_forecast' or self.task_name == 'short_term_forecast':
            dec_out = self.forecast(index, x_enc, x_mark_enc, x_dec, x_mark_dec)
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
