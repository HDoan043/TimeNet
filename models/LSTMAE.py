import torch.nn as nn

class LSTMAE(nn.Module):
    def __init__(self, configs):
        super().__init__()

        self.encoder = nn.LSTM(
            input_size=configs.enc_in,
            hidden_size=configs.d_model,
            num_layers=configs.e_layers,
            batch_first=True
        )

        self.decoder = nn.Linear(configs.d_model, configs.enc_in)

    def forward(self, x):
        # x: [B, T, D]

        out, _ = self.encoder(x)   # [B, T, d_model]

        out = self.decoder(out)    # [B, T, D]

        return out
