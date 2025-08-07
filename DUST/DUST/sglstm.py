import torch
import torch.nn as nn
import torch.nn.functional as F

class SGLSTMCell(nn.Module):
    """
    SGLSTMCell: A custom LSTM cell with some modifications for better sequence modeling.
    """
    def __init__(self, input_dim, hidden_dim, activation="tanh", recurrent_activation="sigmoid"):
        super(SGLSTMCell, self).__init__()
        self.hidden_dim = hidden_dim
        self.activation = getattr(F, activation)
        self.recurrent_activation = getattr(F, recurrent_activation)

        # Weight initialization
        self.Wx = nn.Linear(input_dim, hidden_dim * 4)  # Input mapping (for gates: input, forget, cell, output)
        self.Wh = nn.Linear(hidden_dim, hidden_dim * 4)  # Recurrent mapping (hidden states)
        self.bias = nn.Parameter(torch.zeros(hidden_dim * 4))  # Bias term
        self.W_output = nn.Linear(hidden_dim, hidden_dim)  # Output layer (for candidate hidden states)

    def forward(self, x, hidden):
        h_prev, c_prev = hidden  # Previous states (hidden and cell states)

        # Gate calculations: input gate, forget gate, cell gate, and output gate
        gates = self.Wx(x) + self.Wh(h_prev) + self.bias
        i, f, g, o = gates.chunk(4, dim=-1)

        i = self.recurrent_activation(i)  # Input gate
        f = self.recurrent_activation(f)  # Forget gate
        g = self.activation(g)  # Cell state candidate
        o = self.recurrent_activation(o)  # Output gate

        # Update cell state
        c = f * c_prev + i * g  # Update cell state
        h = o * self.activation(c)  # Hidden state (output)

        return h, (h, c)


class SGLSTMLayer(nn.Module):
    """
    SGLSTMLayer: A layer of stacked SGLSTMCells, which operate on the input sequence.
    """
    def __init__(self, input_dim, hidden_dim, num_layers=3):
        super(SGLSTMLayer, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.cells = nn.ModuleList(
            [SGLSTMCell(input_dim if i == 0 else hidden_dim, hidden_dim) for i in range(num_layers)]
        )

    def forward(self, x):
        batch_size, seq_len, _ = x.shape
        h_t = torch.zeros(batch_size, self.hidden_dim, device=x.device)
        c_t = torch.zeros(batch_size, self.hidden_dim, device=x.device)

        for i, cell in enumerate(self.cells):
            outputs = []
            for t in range(seq_len):
                h_t, (h_t, c_t) = cell(x[:, t, :], (h_t, c_t))  # Process each time step in sequence
                outputs.append(h_t.unsqueeze(1))  # Add hidden states to outputs

            x = torch.cat(outputs, dim=1)  # Update x with outputs for the next layer

        return x
