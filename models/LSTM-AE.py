import torch.nn as nn

class LSTMAE(nn.Module):
    def __init__(self, enc_in, d_model, n_layers):
        super().__init__()

        self.encoder = nn.LSTM(
            input_size=enc_in,
            hidden_size=d_model,
            num_layers=n_layers,
            batch_first=True
        )

        self.decoder = nn.Linear(d_model, enc_in)

    def forward(self, x):
        # x: [B, T, D]

        out, _ = self.encoder(x)   # [B, T, d_model]

        out = self.decoder(out)    # [B, T, D]

        return out
