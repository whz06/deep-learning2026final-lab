"""v8/models/gru_spatial_v8.py — GRU + Industry Embedding + Industry-Gated Spatial Attention.

Features: 31-dim (24 temporal + 7 cross)
Industry: Embedding(n_industries, 8) → concat into head
Spatial: SparseSpatialAttention with same-industry boosting
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import os, sys

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, _ROOT)
from v2.models.gru import GRURanker
from v8.models.spatial_attn_v8 import SparseSpatialAttention


class GRURankerSpatialV8(GRURanker):
    def __init__(self, input_dim=31, hidden_size=128, num_layers=1, dropout=0.2,
                 bidirectional=False, d_proj=32, K=5,
                 n_industries=100, ind_emb_dim=8, lambda_gate=0.1):
        super().__init__(input_dim, hidden_size, num_layers, dropout, bidirectional)
        self.spatial = SparseSpatialAttention(d_model=hidden_size, d_proj=d_proj, K=K)
        self.ind_emb = nn.Embedding(n_industries, ind_emb_dim, padding_idx=None)
        self.lambda_gate = lambda_gate

        head_in = hidden_size + d_proj + ind_emb_dim
        self.head = nn.Sequential(
            nn.Linear(head_in, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, 1),
        )

    def forward(self, x, industry_id=None):
        out, _ = self.gru(x)            # [N, T, H]
        h = out[:, -1, :]               # [N, H]
        c = self.spatial(h, industry_id, self.lambda_gate)  # [N, d_proj]

        combined = torch.cat([h, c], dim=-1)
        if industry_id is not None:
            ind_f = self.ind_emb(industry_id)  # [N, ind_emb_dim]
            combined = torch.cat([combined, ind_f], dim=-1)

        return self.head(combined).squeeze(-1)
