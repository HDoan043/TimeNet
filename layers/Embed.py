import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import weight_norm
import math


class PositionalEmbedding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEmbedding, self).__init__()
        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model).float()
        pe.require_grad = False

        position = torch.arange(0, max_len).float().unsqueeze(1)
        div_term = (torch.arange(0, d_model, 2).float()
                    * -(math.log(10000.0) / d_model)).exp()

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return self.pe[:, :x.size(1)]


class TokenEmbedding(nn.Module):
    def __init__(self, c_in, d_model):
        super(TokenEmbedding, self).__init__()
        padding = 1 if torch.__version__ >= '1.5.0' else 2
        self.tokenConv = nn.Conv1d(in_channels=c_in, out_channels=d_model,
                                   kernel_size=3, padding=padding, 
                                   # padding_mode='circular', 
                                   bias=False)
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(
                    m.weight, mode='fan_in', nonlinearity='leaky_relu')

    def forward(self, x):
        x = self.tokenConv(x.permute(0, 2, 1)).transpose(1, 2)
        return x


class FixedEmbedding(nn.Module):
    def __init__(self, c_in, d_model):
        super(FixedEmbedding, self).__init__()

        w = torch.zeros(c_in, d_model).float()
        w.require_grad = False

        position = torch.arange(0, c_in).float().unsqueeze(1)
        div_term = (torch.arange(0, d_model, 2).float()
                    * -(math.log(10000.0) / d_model)).exp()

        w[:, 0::2] = torch.sin(position * div_term)
        w[:, 1::2] = torch.cos(position * div_term)

        self.emb = nn.Embedding(c_in, d_model)
        self.emb.weight = nn.Parameter(w, requires_grad=False)

    def forward(self, x):
        return self.emb(x).detach()


class TemporalEmbedding(nn.Module):
    def __init__(self, d_model, embed_type='fixed', freq='h', encode_timestamps = ["month", "day", "weekday", "hour"]):
        super(TemporalEmbedding, self).__init__()

        '''
        nn.Embedding will create a learnable table mapping an index i to a vector in hidden space
        So:
            _ To embed month: there are 12 distinctive values of months: 1 - 12
                Use the month directly as the index of the embedding table
                -> the number of rows in nn.Embedding for month should be 13 with index from 0 to 12
                ( index 1 - 12 corresponds to month 1 - 12, the row with index of 0 is not used)
            _ To embed day in month: there are 31 distinctive values of days: 1 - 31
                Use the day directly as the index of the embedding table
                -> the number of rows in nn.Embedding for day should be 32 with index from 0 to 31
            _ To embed hour: there are 24 distinctive values of hour: 0 - 23 
                -> the number of rows in nn.Embedding for hour should be 24
            _ To embed weekday: there are distinctive values of weakday: Monday - Sunday
                Consider they are integer: 0 -> 6
                -> the number of rows in nn.Embedding for weekday should be 7
            _ To embed minutes: there are 60 distinctive values of minute: 0 - 59
                But it is too noisy to embed each minute separately. 
                Instead, minutes are represented with 4 blocks, each block is embeded with only a vector in hidden space:
                + Minute 0  - 14: block 0
                + Minute 15 - 30: block 1
                + Minute 31 - 44: block 2
                + Minute 45 - 59: block 3
                -> the number of rows in nn.Embedding for blocks should be 4
        '''
        if freq == "5min":
            five_min_size = 12
        minute_size = 4
        hour_size = 24
        weekday_size = 7
        day_size = 32
        month_size = 13

        Embed = FixedEmbedding if embed_type == 'fixed' else nn.Embedding
        if freq == 't' or "minute" in encode_timestamps:
            self.minute_embed = Embed(minute_size, d_model)
        if freq == '5min' or "5min" in encode_timestamps:
            self.five_min_embed = Embed(five_min_size, d_model)
        if "hour" in encode_timestamps:
            self.hour_embed = Embed(hour_size, d_model)
        if "weekday" in encode_timestamps:
            self.weekday_embed = Embed(weekday_size, d_model)
        if "day" in encode_timestamps:
            self.day_embed = Embed(day_size, d_model)
        if "month" in encode_timestamps:
            self.month_embed = Embed(month_size, d_model)

    def forward(self, x):
        x = x.long()
        minute_x = self.minute_embed(x[:, :, 4]) if hasattr(
            self, 'minute_embed') else 0
        five_min_x = self.five_min_embed(x[:, :, 4]) if hasattr(
            self, 'five_min_embed') else 0
        hour_x = self.hour_embed(x[:, :, 3]) if hasattr(
            self, 'hour_embed') else 0
        weekday_x = self.weekday_embed(x[:, :, 2]) if hasattr(
            self, 'weekday_embed') else 0
        day_x = self.day_embed(x[:, :, 1]) if hasattr(
            self, 'day_embed') else 0
        month_x = self.month_embed(x[:, :, 0]) if hasattr(
            self, 'month_embed') else 0

        return hour_x + weekday_x + day_x + month_x + minute_x + five_min_x


class TimeFeatureEmbedding(nn.Module):
    def __init__(self, d_model, embed_type='timeF', freq='h'):
        super(TimeFeatureEmbedding, self).__init__()

        freq_map = {'h': 4, 't': 5, 's': 6,
                    'm': 1, 'a': 1, 'w': 2, 'd': 3, 'b': 3}
        d_inp = freq_map[freq]
        self.embed = nn.Linear(d_inp, d_model, bias=False)

    def forward(self, x):
        return self.embed(x)

########## New embedding ##############
class ChannelEmbedding(nn.Module):
    def __init__(self, d_model, c_in):
        super(ChannelEmbedding, self).__init__()
        self.channel_projector = nn.Parameter(torch.randn(c_in, d_model))
        self.channel_bias = nn.Parameter(torch.randn(d_model))
        self.channel_embedding = nn.Linear(in_features = c_in, out_features = 1, bias = True)
        self.activate = nn.Sigmoid()
        self.softmax  = nn.Softmax(dim = -1)
    def forward(self, x, corr_matrix):                                                   # x: [seq_len x d_model], corr_matrix: [c_in x c_in]
        channel_presentation = corr_matrix.matmul(self.channel_projector)                # channel_presentation : [c_in x d_model]
        channel_presentation = channel_presentation + self.channel_bias                  # channel_presentation : [c_in x d_model]
        channel_presentation = self.activate(channel_presentation)                       # channel_presentatin  : [c_in x d_model] 
        channel_presentation = self.softmax(channel_presentation)                        # channel_presentation : [c_in x d_model]
        
        # priori_embedding = self.channel_embedding(channel_presentation)                  # priori_embedding     : [d_model x 1]
        # priori_embedding = self.activate(priori_embedding)                               # priori_embedding     : [d_model x 1]
        # priori_embedding = self.softmax(priori_embedding)                                # priori_embedding     : [d_model x 1]

        return x.matmul(channel_presentation)                                                 # x: [seq_len x d_model]

        # return x + priori_embedding.permute(1,0)                                         # [seq_len x d_model]
        # return x

#######################################################
class DataEmbedding(nn.Module):
    def __init__(self, c_in, d_model, embed_type='fixed', freq='h', dropout=0.1, encode_timestamps  = ["month", "day", "weekday", "hour"] ):
        super(DataEmbedding, self).__init__()

        self.value_embedding = TokenEmbedding(c_in=c_in, d_model=d_model)
        self.position_embedding = PositionalEmbedding(d_model=d_model)
        self.temporal_embedding = TemporalEmbedding(d_model=d_model, embed_type=embed_type,
                                                    freq=freq, encode_timestamps = encode_timestamps)  if embed_type != 'timeF' \
                                                else TimeFeatureEmbedding(d_model=d_model, embed_type=embed_type, freq=freq)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x, x_mark):
        if x_mark is None:
            x = self.value_embedding(x) + self.position_embedding(x)
        else:
            x = self.value_embedding(
                x) + self.temporal_embedding(x_mark) + self.position_embedding(x)
        return self.dropout(x)

class PrioriDataEmbedding(nn.Module):
    def __init__(self, c_in, d_model, embed_type='fixed', freq='h', dropout=0.1):
        super(PrioriDataEmbedding, self).__init__()

        self.value_embedding = TokenEmbedding(c_in=c_in, d_model=d_model)
        self.position_embedding = PositionalEmbedding(d_model=d_model)
        self.temporal_embedding = TemporalEmbedding(d_model=d_model, embed_type=embed_type,
                                                    freq=freq) if embed_type != 'timeF' else TimeFeatureEmbedding(
            d_model=d_model, embed_type=embed_type, freq=freq)
        self.dropout = nn.Dropout(p=dropout)
        self.local_channel_embedding = ChannelEmbedding(d_model, c_in)
        self.global_channel_embedding= ChannelEmbedding(d_model, c_in)

    def forward(self, x, x_mark, corr_matrix):                                    # x: [batch_size x seq_len x c_in], corr_matrix: [c_in x c_in]
        ######## Calculate correlation matrix of a sample ########
        local_corr_matrix_ls = []
        for matrix in x:                                                           # matrix: [seq_len x c_in]
            local_corr_matrix = torch.corrcoef(matrix.permute(1,0))                # local_corr_matrix : [ c_in x c_in ]
            local_corr_matrix = torch.nan_to_num(local_corr_matrix, nan=0.0)       # local_corr_matrix : [ c_in x c_in ]
            local_corr_matrix_ls.append(local_corr_matrix)                         # local_corr_matrix_ls: [ batch_size * [c_in x c_in] ]

        ######## Global channel embeding ########
        # Process nan
        corr_matrix = torch.nan_to_num(corr_matrix, nan=0.0)
        global_channel_embed = []
        for sample in x:                                                                    # sample: [seq_len x c_in]
            global_channel_embed.append(self.global_channel_embedding(sample, corr_matrix)) # global_channel_embed: [ batch_size * [seq_len x d_model] ]
        global_channel_embed = torch.stack(global_channel_embed, dim=0)                     # global_channel_embed: [batch_size x seq_len x d_model]
        
        ######## Local  channel embedding ########
        local_channel_embed = []
        for i in range(x.shape[0]):
            sample = x[i]                                                                       # sample: [seq_len x c_in]
            local_corr_matrix = local_corr_matrix_ls[i]                                         # local_corr_matrix: [c_in x c_in]
            local_channel_embeded = self.local_channel_embedding(sample, local_corr_matrix)
            local_channel_embed.append(local_channel_embeded)                                   # local_channel_embed: [batch_size * [seq_len x d_model] } 
        local_channel_embed = torch.stack(local_channel_embed, dim = 0)                         # local_channel_embed: [batch_size x seq_len x d_model]
        
        ######## Token Embedding and Positional Embedding ########
        if x_mark is None:
            x = self.value_embedding(x) + self.position_embedding(x)                # x: [batch_size x seq_len x d_model]
        else:
            x = self.value_embedding(
                x) + self.temporal_embedding(x_mark) + self.position_embedding(x)   # x: [batch_size x seq_len x d_model]

        
        ######## Combind embedding ############
        x = 0.5*x  + 0.25*local_channel_embed + 0.25*global_channel_embed
        x = self.dropout(x)
        return x
    
class DataEmbedding_inverted(nn.Module):
    def __init__(self, c_in, d_model, embed_type='fixed', freq='h', dropout=0.1):
        super(DataEmbedding_inverted, self).__init__()
        self.value_embedding = nn.Linear(c_in, d_model)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x, x_mark):
        x = x.permute(0, 2, 1)
        # x: [Batch Variate Time]
        if x_mark is None:
            x = self.value_embedding(x)
        else:
            x = self.value_embedding(torch.cat([x, x_mark.permute(0, 2, 1)], 1))
        # x: [Batch Variate d_model]
        return self.dropout(x)


class DataEmbedding_wo_pos(nn.Module):
    def __init__(self, c_in, d_model, embed_type='fixed', freq='h', dropout=0.1, encode_timestamps  = ["month", "day", "weekday", "hour"] ):
        super(DataEmbedding_wo_pos, self).__init__()

        self.value_embedding = TokenEmbedding(c_in=c_in, d_model=d_model)
        self.position_embedding = PositionalEmbedding(d_model=d_model)
        self.temporal_embedding = TemporalEmbedding(d_model=d_model, embed_type=embed_type,
                                                    freq=freq, encode_timestamps = encode_timestamps) if embed_type != 'timeF' \
                                                    else TimeFeatureEmbedding(d_model=d_model, embed_type=embed_type, freq=freq)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x, x_mark):
        if x_mark is None:
            x = self.value_embedding(x)
        else:
            x = self.value_embedding(x) + self.temporal_embedding(x_mark)
        return self.dropout(x)


class PatchEmbedding(nn.Module):
    def __init__(self, d_model, patch_len, stride, padding, dropout):
        super(PatchEmbedding, self).__init__()
        # Patching
        self.patch_len = patch_len
        self.stride = stride
        self.padding_patch_layer = nn.ReplicationPad1d((0, padding))

        # Backbone, Input encoding: projection of feature vectors onto a d-dim vector space
        self.value_embedding = nn.Linear(patch_len, d_model, bias=False)

        # Positional embedding
        self.position_embedding = PositionalEmbedding(d_model)

        # Residual dropout
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # do patching
        n_vars = x.shape[1]
        x = self.padding_patch_layer(x)
        x = x.unfold(dimension=-1, size=self.patch_len, step=self.stride)
        x = torch.reshape(x, (x.shape[0] * x.shape[1], x.shape[2], x.shape[3]))
        # Input encoding
        x = self.value_embedding(x) + self.position_embedding(x)
        return self.dropout(x), n_vars
