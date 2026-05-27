import torch
import torch.nn as nn
import numpy as np
from models import Autoformer, Transformer, TimesNet, Nonstationary_Transformer, DLinear, FEDformer, \
    Informer, LightTS, Reformer, ETSformer, Pyraformer, PatchTST, MICN, Crossformer, FiLM, iTransformer, \
    Koopa, TiDE, FreTS, TimeMixer, TSMixer, SegRNN, MambaSimple, TemporalFusionTransformer, SCINet, PAttn, TimeXer, \
    WPMixer, MultiPatchFormer, KANAD, MSGNet, TimeFilter, TimesNet_update_v1, TimesNet_update_v2, LSTMAE

from layers.Corrector import *

class Model(nn.Module):
    def __init__(self, configs):
        super(Model, self).__init__()
        self.model_dict = {
            'LSTMAE': LSTMAE,
            'TimesNet': TimesNet,
            'Autoformer': Autoformer,
            'Transformer': Transformer,
            'Nonstationary_Transformer': Nonstationary_Transformer,
            'DLinear': DLinear,
            'FEDformer': FEDformer,
            'Informer': Informer,
            'LightTS': LightTS,
            'Reformer': Reformer,
            'ETSformer': ETSformer,
            'PatchTST': PatchTST,
            'Pyraformer': Pyraformer,
            'MICN': MICN,
            'Crossformer': Crossformer,
            'FiLM': FiLM,
            'iTransformer': iTransformer,
            'Koopa': Koopa,
            'TiDE': TiDE,
            'FreTS': FreTS,
            'MambaSimple': MambaSimple,
            'TimeMixer': TimeMixer,
            'TSMixer': TSMixer,
            'SegRNN': SegRNN,
            'TemporalFusionTransformer': TemporalFusionTransformer,
            "SCINet": SCINet,
            'PAttn': PAttn,
            'TimeXer': TimeXer,
            'WPMixer': WPMixer,
            'MultiPatchFormer': MultiPatchFormer,
            'KANAD': KANAD,
            'MSGNet': MSGNet,
            'TimeFilter': TimeFilter,
            'TimesNetv1': TimesNet_update_v1,
            'TimesNetv2': TimesNet_update_v2
        }
        self.configs = configs
        
        self.base_model = self.model_dict[configs.base_model].Model(configs).float()
        # load checkpoint
        checkpoint = torch.load(configs.base_model_checkpoint)
        
        # Tạo một dictionary mới để gọt bỏ chữ 'module.'
        from collections import OrderedDict
        new_state_dict = OrderedDict()
        for k, v in checkpoint.items():
            name = k[7:] if k.startswith('module.') else k # Bỏ 7 ký tự đầu ('module.')
            new_state_dict[name] = v
            
        # Load lại dict đã gọt dũa
        self.base_model.load_state_dict(new_state_dict)
        
        # freeze base_model
        for p in self.base_model.parameters():
            p.requires_grad = False
        
        self.reconstructor = nn.MSELoss(reduction='none')

        teacher_d_model = configs.d_model if configs.use_teacher_hidden == 1 else None

        if configs.corrector.lower() in ["lstm", "lstm_corrector", "lstm-corrector"]:
            self.corrector = LSTM_Corrector(
                configs.enc_in, configs.corrector_d_model, configs.corrector_layers, teacher_d_model, apply_causal = configs.apply_causal)
        else: 
            self.corrector = TCN_Corrector(
                configs.enc_in, configs.corrector_d_model, configs.corrector_layers, 
                configs.corrector_kernel_size, teacher_d_model, apply_causal = configs.apply_causal
            )
        
        
    def forward(self, x, x_mark_enc, x_dec, x_mark_dec):        # [B,win_size,channels]
        # get the hidden state and the score from base model
        hidden_state, dec_out = self.base_model(x, x_mark_enc, x_dec, x_mark_dec)
        
        # calculate reconstruct score
        reconstruct_score = self.reconstructor(x, dec_out)                      # [B, win_size, channels]
        reconstruct_score = reconstruct_score.mean(dim=2, keepdim=True)         # [B, win_size, 1]

        # MỚI: Tự động Normalize base_score trong từng cửa sổ (Instance Norm)
        # Giúp base_score có mean=0, std=1.
        score_mean = reconstruct_score.mean(dim=1, keepdim=True)
        score_std = reconstruct_score.std(dim=1, keepdim=True) + 1e-5
        reconstruct_score_norm = (reconstruct_score - score_mean) / score_std
        
        # correction (Truyền bản Norm vào)
        if self.configs.use_teacher_hidden == 1:
            final_score, delta = self.corrector(x, reconstruct_score_norm, hidden_state)
        else:
            final_score, delta = self.corrector(x, reconstruct_score_norm)
        
        return final_score                                                      # [B, win_size]
