"""v2/models/gru.py — GRU ranker (same as v1, accepts 22-dim input)."""
import torch
import torch.nn as nn


class GRURanker(nn.Module):
    def __init__(self, input_dim: int, hidden_size: int = 128, num_layers: int = 2,
                 dropout: float = 0.1, bidirectional: bool = False):
        super().__init__()
        self.gru = nn.GRU(
            input_size=input_dim, hidden_size=hidden_size,
            num_layers=num_layers, batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=bidirectional,
        )
        dir_mult = 2 if bidirectional else 1
        self.head = nn.Sequential(
            nn.Linear(hidden_size * dir_mult, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, 1),
        )

    def forward(self, x):
        out, _ = self.gru(x)
        last = out[:, -1, :]
        return self.head(last).squeeze(-1)
