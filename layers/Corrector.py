import torch
import torch.nn as nn
import numpy as np

class LSTM_Corrector(nn.Module):
    def __init__(self, raw_dim, d_model=128, lstm_layers=1, teacher_d_model=None, apply_causal=False):
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
            bidirectional=not apply_causal
        )
        
        # Đầu ra của Bi-LSTM sẽ có số chiều là: d_model * 2 (Vì có 2 chiều)
        # Ta dùng một lớp Linear nhỏ để nén nó về 1 giá trị Delta
        lstm_channels = d_model * 2 if not apply_causal else d_model
        hidden_d_model = d_model if not apply_causal else d_model//2
        self.projector = nn.Sequential(
            nn.Linear(lstm_channels, hidden_d_model),
            nn.ReLU(),
            nn.Linear(hidden_d_model, 1) # Xuất ra 1 giá trị (Delta) cho mỗi timestamp
        )
        
    def forward(self, raw_x, base_score, teacher_hidden_state=None):
        """
        raw_x: [B, win_size, raw_dim] (Dữ liệu 5G gốc)
        base_score: [B, win_size, 1] (Điểm do mô hình gốc phán)
        teacher_hidden_state: [B, win_size, d_model] (Không gian ẩn của mô hình pretrain)
        """
        
        # 1. Gộp tất cả thông tin lại làm Đầu vào
        # Shape sau khi nối: [B, win_size, input_dim]
        if not isinstance(teacher_hidden_state, torch.Tensor):
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

class TemporalBlock(nn.Module):
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, dropout=0.2, apply_causal=False):
        super(CausalTemporalBlock, self).__init__()

        # TÍNH PADDING: Đủ để kernel nhìn xa mà không làm tụt độ dài
        self.pad_size = (kernel_size - 1) * dilation
        self.apply_causal = apply_causal

        if apply_causal:
            self.conv1 = nn.Conv1d(n_inputs, n_outputs, kernel_size, stride=stride, padding=0, dilation=dilation)
            self.conv2 = nn.Conv1d(n_outputs, n_outputs, kernel_size, stride=stride, padding=0, dilation=dilation)
        else:
            self.conv1 = nn.Conv1d(n_inputs, n_outputs, kernel_size, stride=stride, padding=self.pad_size//2, dilation=dilation)
            self.conv2 = nn.Conv1d(n_outputs, n_outputs, kernel_size, stride=stride, padding=self.pad_size//2, dilation=dilation)
            
        self.relu1 = nn.ReLU()
        self.relu2 = nn.ReLU()
        
        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.relu = nn.ReLU()

    def forward(self, x):
        res = x if self.downsample is None else self.downsample(x)
        
        # QUAN TRỌNG NHẤT: Padding lệch trái (Nhìn quá khứ)
        if not self.apply_causal:
            out = F.pad(x, (self.pad_size, 0)) # Thêm 0 vào trái, không thêm gì vào phải
        out = self.relu1(self.conv1(out))

        if not self.apply_causal:
            out = F.pad(out, (self.pad_size, 0))
        out = self.relu2(self.conv2(out))
        
        return self.relu(out + res)
    
class TCN_Corrector(nn.Module):
    def __init__(self, raw_dim, d_model, layers, kernel_size = 12, teacher_d_model=None, apply_causal=False):
        super(TCN_Corrector, self).__init__()
        input_dim = raw_dim + 1 + (0 if not isinstance(teacher_d_model, int) else teacher_d_model)

        num_channels = list(np.round(np.linespace(input_dim, d_model, layers+1)))
        num_channels = num_channels[1:]
        layers = []
        for i in range(len(num_channels)):
            dilation_size = 2 ** i # Giãn nở: 1, 2, 4...
            in_channels = input_dim if i == 0 else num_channels[i-1]
            out_channels = num_channels[i]
            
            layers.append(TemporalBlock(in_channels, out_channels, kernel_size, 1, dilation_size, dropout, apply_causal))

        self.tcn = nn.Sequential(*layers)
        self.projector = nn.Linear(num_channels[-1], 1)

    def forward(self, raw_x, base_score):
        combined = torch.cat([raw_x, base_score], dim=-1)
        combined = combined.transpose(1, 2)  # Đưa Channel lên giữa cho Conv1d
        
        tcn_out = self.tcn(combined)         # Chạy qua TCN
        tcn_out = tcn_out.transpose(1, 2)    # Đưa Length về lại giữa
        
        delta = self.projector(tcn_out)
        
        final_score = base_score + delta
        return final_score.squeeze(-1), delta
        
