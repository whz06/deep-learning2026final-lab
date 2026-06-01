"""
v1/model.py — GRU-based stock ranker.

Input:  [N, T, F]  — N stocks, T time-steps, F features
Output: [N]        — scalar ranking score per stock
"""
import torch
import torch.nn as nn


class GRURanker(nn.Module):
    def __init__(self, input_dim: int, hidden_size: int = 128, num_layers: int = 2,
                 dropout: float = 0.3, bidirectional: bool = False):
        super().__init__()
        self.gru = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
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
        out, _ = self.gru(x)          # [N, T, H*D]
        last = out[:, -1, :]          # [N, H*D] — final time-step
        score = self.head(last)       # [N, 1]
        return score.squeeze(-1)      # [N]
