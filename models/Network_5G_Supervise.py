import torch
import torch.nn as nn
from models import Autoformer, Transformer, TimesNet, Nonstationary_Transformer, DLinear, FEDformer, \
    Informer, LightTS, Reformer, ETSformer, Pyraformer, PatchTST, MICN, Crossformer, FiLM, iTransformer, \
    Koopa, TiDE, FreTS, TimeMixer, TSMixer, SegRNN, MambaSimple, TemporalFusionTransformer, SCINet, PAttn, TimeXer, \
    WPMixer, MultiPatchFormer, KANAD, MSGNet, TimeFilter, TimesNet_update_v1, TimesNet_update_v2, LSTMAE

class Corrector(nn.Module):
    def __init__(self, raw_dim, d_model=128, lstm_layers=1, teacher_d_model=None):
        super(Corrector, self).__init__()
        
        # Tổng số chiều đầu vào = Số biến gốc + 1 (Điểm Base_Score)
        if not teacher_d_model:
            input_dim = raw_dim + 1
        else:
            input_dim = raw_dim + teacher_d_model + 1
        
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
        
    def forward(self, raw_x, base_score, teacher_hidden_state=None):
        """
        raw_x: [B, win_size, raw_dim] (Dữ liệu 5G gốc)
        base_score: [B, win_size, 1] (Điểm do mô hình gốc phán)
        teacher_hidden_state: [B, win_size, d_model] (Không gian ẩn của mô hình pretrain)
        """
        
        # 1. Gộp tất cả thông tin lại làm Đầu vào
        # Shape sau khi nối: [B, win_size, input_dim]
        if not teacher_hidden_state:
            combined_input = torch.cat([raw_x, base_score], dim=-1)
        else:
            combined_input = torch.cat([raw_x, teacher_hidden_state, base_score], dim=-1)
            
        # 2. Đưa qua Bi-LSTM
        # lstm_out shape: [B, win_size, d_model * 2]
        lstm_out, (h_n, c_n) = self.bilstm(combined_input)
        
        # 3. Phóng nó ra thành Delta Score
        # delta shape: [B, win_size, 1]
        delta = self.projector(lstm_out)
        
        # 4. Tính Final Score
        # Có thể dùng hàm Tanh để kẹp Delta trong khoảng [-1, 1] (Sửa tối đa 1 điểm)
        # delta_clipped = torch.tanh(delta) 
        final_score = base_score + delta                # (-inf, +inf)
        
        final_score = final_score.squeeze()             # [B, win_size]
        
        return final_score, delta
    
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
        # corrector
        self.corrector = Corrector(
            configs.enc_in, configs.corrector_d_model, configs.corrector_layers, teacher_d_model)
        
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
