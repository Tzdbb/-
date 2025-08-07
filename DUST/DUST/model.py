import torch
import torch.nn as nn
import math
from feature_extraction import FeatureReductionModule

class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.d_model = d_model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, _ = x.size()
        position = torch.arange(seq_len, device=x.device).unsqueeze(0)
        div_term = torch.exp(torch.arange(0, self.d_model, 2, device=x.device).float() *
                             (-math.log(10000.0) / self.d_model))
        pe = torch.zeros(batch_size, seq_len, self.d_model, device=x.device)
        pe[..., 0::2] = torch.sin(position * div_term)
        pe[..., 1::2] = torch.cos(position * div_term)

        x = x + pe
        return self.dropout(x)

class DualDomainFusionTransformer(nn.Module):
    def __init__(self, time_dim, freq_dim, d_model=128, nhead=4, num_layers=4, dim_feedforward=256, dropout=0.3):
        super().__init__()
        self.time_reducer = FeatureReductionModule(time_dim, d_model // 4)
        self.freq_reducer = FeatureReductionModule(freq_dim, d_model // 4)
        self.fc_combine = nn.Linear(freq_dim + d_model // 4 + d_model // 4 + time_dim, d_model)
        self.pos_encoder = PositionalEncoding(d_model, dropout)
        encoder_layers = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward, dropout=dropout)
        self.transformer = nn.TransformerEncoder(encoder_layers, num_layers=num_layers)

    def forward(self, time_input, freq_input):
        p1 = self.freq_bn(freq_input)
        p2 = self.freq_reducer(freq_input)
        time_fft = torch.fft.fft(time_input, dim=1)
        time_fft_abs = torch.abs(time_fft)
        p3 = self.time_reducer(time_fft_abs)
        p4 = self.time_bn(time_input)

        combined = torch.cat([p1, p2, p3, p4], dim=1)
        fused = self.fc_combine(combined)
        src = fused.unsqueeze(1)
        src = self.pos_encoder(src)
        trans_out = self.transformer(src)
        output = self.classifier(trans_out.squeeze(1))
        return output
