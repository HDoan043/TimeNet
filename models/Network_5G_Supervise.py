import torch
import torch.nn as nn
from models import Autoformer, Transformer, TimesNet, Nonstationary_Transformer, DLinear, FEDformer, \
    Informer, LightTS, Reformer, ETSformer, Pyraformer, PatchTST, MICN, Crossformer, FiLM, iTransformer, \
    Koopa, TiDE, FreTS, TimeMixer, TSMixer, SegRNN, MambaSimple, TemporalFusionTransformer, SCINet, PAttn, TimeXer, \
    WPMixer, MultiPatchFormer, KANAD, MSGNet, TimeFilter, TimesNet_update_v1, TimesNet_update_v2, LSTMAE

class Corrector(nn.Module):
    def __init__(self, raw_dim, hidden_dim_timesnet, d_model=128, lstm_layers=1):
        super(Corrector, self).__init__()
        
        # Tổng số chiều đầu vào = Số biến gốc + Số chiều Latent của TimesNet + 1 (Điểm Base_Score)
        input_dim = raw_dim + hidden_dim_timesnet + 1
        
        # Khai báo Bi-LSTM siêu gọn nhẹ
        # batch_first=True nghĩa là Input shape phải là [Batch, Seq_len, Input_dim]
        self.bilstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=d_model,
            num_layers=lstm_layers,
            batch_first=True,
            bidirectional=True # BI-LSTM chính là nhờ cái cờ này!
        )
        
        # Đầu ra của Bi-LSTM sẽ có số chiều là: d_model * 2 (Vì có 2 chiều)
        # Ta dùng một lớp Linear nhỏ để nén nó về 1 giá trị Delta
        self.projector = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.ReLU(),
            nn.Linear(d_model, 1) # Xuất ra 1 giá trị (Delta) cho mỗi timestamp
        )
        
    def forward(self, raw_x, timesnet_latent, base_score):
        """
        raw_x: [B, win_size, raw_dim] (Dữ liệu 5G gốc)
        timesnet_latent: [B, win_size, hidden_dim_timesnet] (Não của mô hình gốc)
        base_score: [B, win_size, 1] (Điểm do mô hình gốc phán)
        """
        
        # 1. Gộp tất cả thông tin lại làm Đầu vào
        # Shape sau khi nối: [B, win_size, input_dim]
        combined_input = torch.cat([raw_x, timesnet_latent, base_score], dim=-1)
        
        # 2. Đưa qua Bi-LSTM
        # lstm_out shape: [B, win_size, d_model * 2]
        lstm_out, (h_n, c_n) = self.bilstm(combined_input)
        
        # 3. Phóng nó ra thành Delta Score
        # delta shape: [B, win_size, 1]
        delta = self.projector(lstm_out)
        
        # 4. Tính Final Score
        # Có thể dùng hàm Tanh để kẹp Delta trong khoảng [-1, 1] (Sửa tối đa 1 điểm)
        delta_clipped = torch.tanh(delta) 
        final_score = base_score + delta_clipped        # (-inf, +inf)
        
        final_score = final_score.squeeze()             # [B, win_size]
        
        return final_score, delta_clipped
    
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
        
        # corrector
        self.corrector = Corrector(
            configs.enc_in, configs.d_model, configs.corrector_d_model, configs.corrector_layers)
        
    def forward(self, x, x_mark_enc, x_dec, x_mark_dec):        # [B,win_size,channels]
        # get the hidden state and the score from base model
        hidden_state, dec_out = self.base_model(x, x_mark_enc, x_dec, x_mark_dec)
        
        # calculate reconstruct score
        reconstruct_score = self.reconstructor(x, dec_out)                      # [B, win_size, channels]
        reconstruct_score = reconstruct_score.mean(dim=2, keepdim=True)         # [B, win_size, 1]
        
        # correction
        final_score, delta_clipped = self.corrector(x, hidden_state, reconstruct_score)
        
        return final_score                                                      # [B, win_size]
