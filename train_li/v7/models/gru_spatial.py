"""v7/models/gru_spatial.py — GRU + SparseSpatialAttention concat fusion (for v7 T+1 training)."""
import torch, torch.nn as nn, torch.nn.functional as F, sys, os

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(_ROOT, "v2"))
from models.gru import GRURanker
from v7.models.spatial_attn import SparseSpatialAttention

class GRURankerSpatial(GRURanker):
    def __init__(self, input_dim, hidden_size=128, num_layers=1, dropout=0.2,
                 bidirectional=False, d_proj=32, K=5):
        super().__init__(input_dim, hidden_size, num_layers, dropout, bidirectional)
        self.spatial = SparseSpatialAttention(d_model=hidden_size, d_proj=d_proj, K=K)
        self.head = nn.Sequential(
            nn.Linear(hidden_size + d_proj, hidden_size // 2),
            nn.ReLU(), nn.Dropout(dropout), nn.Linear(hidden_size // 2, 1))

    def forward(self, x):
        out, _ = self.gru(x)
        h = out[:, -1, :]
        c = self.spatial(h)
        return self.head(torch.cat([h, c], dim=-1)).squeeze(-1)
