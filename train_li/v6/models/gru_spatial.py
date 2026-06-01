"""v6/models/gru_spatial.py — GRU + SparseSpatialAttention (Step 3).

Supports optional temporal attention pooling (use_attn_pool=True).
If use_attn_pool: GRU → attn pool over time → spatial attention → head.
If not: GRU last hidden → spatial attention → head.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import sys, os

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(_ROOT, "v2"))
from models.gru import GRURanker

from v6.models.spatial_attn import SparseSpatialAttention


class GRURankerSpatial(GRURanker):
    def __init__(self, input_dim, hidden_size=128, num_layers=1, dropout=0.2,
                 bidirectional=False, K=10, use_attn_pool=False):
        super().__init__(input_dim, hidden_size, num_layers, dropout, bidirectional)
        self.use_attn_pool = use_attn_pool
        if use_attn_pool:
            self.temporal_attn = nn.Linear(hidden_size, 1)
        self.spatial_attn = SparseSpatialAttention(d_model=hidden_size, K=K)

    def forward(self, x):
        out, _ = self.gru(x)                     # [N, T, H]
        if self.use_attn_pool:
            w = F.softmax(self.temporal_attn(out), dim=1)
            h = (out * w).sum(dim=1)             # [N, H]
        else:
            h = out[:, -1, :]                    # [N, H]
        h = self.spatial_attn(h)                 # [N, H]
        return self.head(h).squeeze(-1)
