import torch
import torch.nn as nn

class FeatureReductionModule(nn.Module):
    """
    Feature reduction module (similar to Conv1d -> BatchNorm -> MaxPooling -> ReLU).
    """
    def __init__(self, input_dim, output_dim=128):
        super().__init__()
        self.pooled_dim = (input_dim + 1) // 2  # Ceiling value for pooling dimension

        self.conv = nn.Conv1d(in_channels=1, out_channels=64, kernel_size=3, padding=1)
        self.bn = nn.BatchNorm1d(64)
        self.pool = nn.MaxPool1d(kernel_size=2, stride=2)
        self.relu = nn.ReLU()
        self.fc = nn.Linear(64 * self.pooled_dim, output_dim)

    def forward(self, x):
        """
        x shape: [batch_size, seq_len]
        """
        x = x.unsqueeze(1)  # Add channel dimension: [batch_size, 1, seq_len]
        x = self.conv(x)  # [batch_size, 64, seq_len]
        x = self.bn(x)
        x = self.pool(x)  # [batch_size, 64, seq_len // 2]
        x = self.relu(x)
        x = x.view(x.size(0), -1)  # Flatten: [batch_size, 64 * (seq_len // 2)]
        x = self.fc(x)  # [batch_size, output_dim]
        return x

class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.d_model = d_model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x shape: [batch_size, seq_len, embedding_dim]
        """
        batch_size, seq_len, _ = x.size()
        position = torch.arange(seq_len, device=x.device).unsqueeze(0)
        div_term = torch.exp(torch.arange(0, self.d_model, 2, device=x.device).float() *
                             (-math.log(10000.0) / self.d_model))

        pe = torch.zeros(batch_size, seq_len, self.d_model, device=x.device)
        pe[..., 0::2] = torch.sin(position * div_term)
        pe[..., 1::2] = torch.cos(position * div_term)

        x = x + pe
        return self.dropout(x)
