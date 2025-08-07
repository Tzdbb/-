import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import torch.optim as optim


class SGLSTMCell(nn.Module):
    """
    自定义 SGLSTM 单元，用于时序数据处理。
    """

    def __init__(self, input_dim, hidden_dim, activation="tanh", recurrent_activation="sigmoid"):
        super(SGLSTMCell, self).__init__()
        self.hidden_dim = hidden_dim
        self.activation = getattr(F, activation)
        self.recurrent_activation = getattr(F, recurrent_activation)

        # 初始化权重
        self.Wx = nn.Linear(input_dim, hidden_dim * 4)  # 输入映射
        self.Wh = nn.Linear(hidden_dim, hidden_dim * 4)  # 递归映射
        self.bias = nn.Parameter(torch.zeros(hidden_dim * 4))  # 偏置项

    def forward(self, x, hidden):
        h_prev, c_prev = hidden  # 过去的状态

        # 计算输入门、遗忘门、候选状态、输出门
        gates = self.Wx(x) + self.Wh(h_prev) + self.bias
        i, f, c, o = gates.chunk(4, dim=-1)

        i = self.recurrent_activation(i)  # 输入门
        f = self.recurrent_activation(f)  # 遗忘门
        o = self.recurrent_activation(o)  # 输出门

        # 更新细胞状态和隐状态
        c = f * c_prev + i * self.activation(c)  # 细胞状态
        h = o * self.activation(c)  # 隐状态

        return h, (h, c)


class SGLSTMLayer(nn.Module):
    """
    SGLSTM 层：多层 SGLSTM 单元的堆叠
    """

    def __init__(self, input_dim, hidden_dim, num_layers=3):
        super(SGLSTMLayer, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.cells = nn.ModuleList(
            [SGLSTMCell(input_dim if i == 0 else hidden_dim, hidden_dim) for i in range(num_layers)])

    def forward(self, x):
        batch_size, seq_len, _ = x.shape
        h_t = torch.zeros(batch_size, self.hidden_dim, device=x.device)
        c_t = torch.zeros(batch_size, self.hidden_dim, device=x.device)

        for i, cell in enumerate(self.cells):
            outputs = []
            for t in range(seq_len):
                h_t, (h_t, c_t) = cell(x[:, t, :], (h_t, c_t))
                outputs.append(h_t.unsqueeze(1))
            x = torch.cat(outputs, dim=1)  # 更新 x 作为下一个层的输入

        return x


class PositionalEncoding(nn.Module):
    """
    位置编码模块：用于在输入中加入序列的位置信息
    """

    def __init__(self, d_model: int, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.d_model = d_model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        参数:
        x: Tensor, 形状为 [batch_size, seq_len, embedding_dim]
        """
        batch_size, seq_len, _ = x.size()

        # 动态生成位置编码（兼容不同序列长度）
        position = torch.arange(seq_len, device=x.device).unsqueeze(0)
        div_term = torch.exp(
            torch.arange(0, self.d_model, 2, device=x.device).float() *
            (-math.log(10000.0) / self.d_model)
        )

        pe = torch.zeros(batch_size, seq_len, self.d_model, device=x.device)
        pe[..., 0::2] = torch.sin(position * div_term)
        pe[..., 1::2] = torch.cos(position * div_term)

        # 与原输入相加
        x = x + pe
        return self.dropout(x)


class TransformerModel(nn.Module):
    """
    SGLSTM Transformer 模型：用于时序数据的分类模型
    """

    def __init__(self, input_dim, d_model=128, nhead=4, num_layers=4, dim_feedforward=256, dropout=0.3):
        super(TransformerModel, self).__init__()

        # 输入适配层（将输入维度转化为 d_model）
        self.input_adapter = nn.Sequential(
            nn.Linear(in_features=input_dim, out_features=d_model),
            nn.LayerNorm(d_model),
            nn.Dropout(p=dropout)
        )

        # 位置编码
        self.pos_encoder = PositionalEncoding(d_model, dropout)

        # SGLSTM 层
        self.sglstm_layer = SGLSTMLayer(input_dim=d_model, hidden_dim=d_model, num_layers=3)

        # 分类头
        self.classifier = nn.Sequential(
            nn.BatchNorm1d(d_model),
            nn.Dropout(dropout),
            nn.Linear(d_model, 1)
        )

    def forward(self, x):
        """
        模型前向传播

        参数:
        x: 输入张量，形状为 [batch_size, seq_len, input_dim]
        """
        # 输入适配层
        x = self.input_adapter(x)

        # 添加位置编码
        x = self.pos_encoder(x)  # [batch, seq_len, d_model]

        # SGLSTM 层
        sglstm_out = self.sglstm_layer(x)  # [batch, seq_len, d_model]

        # 取序列最后一个时间步的输出进行分类
        output = self.classifier(sglstm_out[:, -1, :])  # 取最后一个时间步的输出，形状为 [batch_size, 1]

        return output


if __name__ == '__main__':
    # 示例：如何实例化和使用模型
    # 假设输入数据为 [batch_size, seq_len, input_dim]
    input_dim = 64  # 输入维度（特征数）
    seq_len = 100  # 序列长度
    batch_size = 32

    model = TransformerModel(input_dim=input_dim)

    # 随机生成输入数据
    x = torch.randn(batch_size, seq_len, input_dim)

    # 模型前向传播
    output = model(x)

    # 打印输出形状
    print(f"Model output shape: {output.shape}")  # 应该是 [batch_size, 1]
