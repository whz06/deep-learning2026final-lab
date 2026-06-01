"""v6/models/gru_attn.py — GRU + learnable temporal attention pooling (Step 4).

Replaces GRU last-hidden with a softmax-weighted sum over all timesteps.
Adds only 129 parameters (Linear(128, 1)).
If attention degenerates to focus on t=-1, this is equivalent to baseline GRU.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import sys, os

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(_ROOT, "v2"))
from models.gru import GRURanker


class GRURankerAttn(GRURanker):
    def __init__(self, input_dim, hidden_size=128, num_layers=1, dropout=0.2,
                 bidirectional=False):
        super().__init__(input_dim, hidden_size, num_layers, dropout, bidirectional)
        self.temporal_attn = nn.Linear(hidden_size, 1)

    def forward(self, x):
        out, _ = self.gru(x)                # [N, T, H]
        w = self.temporal_attn(out)         # [N, T, 1]
        w = F.softmax(w, dim=1)             # normalize over time
        pooled = (out * w).sum(dim=1)       # [N, H]
        return self.head(pooled).squeeze(-1)
