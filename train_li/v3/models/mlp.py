"""v2/models/mlp.py — MLP ranker: flatten time series → FC stack → score."""
import torch
import torch.nn as nn


class MLPRanker(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 512,
                 n_layers: int = 3, dropout: float = 0.3):
        super().__init__()
        layers = []
        in_dim = input_dim
        for i in range(n_layers):
            layers.append(nn.Linear(in_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.Dropout(dropout))
            in_dim = hidden_dim
        layers.append(nn.Linear(hidden_dim, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        # x: [N, T, F] → flatten to [N, T*F]
        x = x.reshape(x.shape[0], -1)
        return self.net(x).squeeze(-1)
