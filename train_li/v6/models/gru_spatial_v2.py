"""v6/models/gru_spatial_v2.py — GRU + SparseSpatialAttention with 3 fusion methods.

Phase 1 sweep per SPATIAL_ATTENTION_PLAN.md:
  S1 (concat):  [h(128) ‖ context(32)] → Linear(160→64)→ReLU→Dropout→Linear(64→1)
  S2 (gated):   gate = σ(Linear(160→1)), fused = gate*h + (1-gate)*proj(context)
  S3 (residual): h + proj(context), proj = Linear(32→128)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import sys, os

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(_ROOT, "v2"))
from models.gru import GRURanker

from v6.models.spatial_attn import SparseSpatialAttention


class GRURankerSpatialConcat(GRURanker):
    """S1: concat [h ‖ context] → head. Head dim = hidden_size + d_proj → ... → 1."""
    def __init__(self, input_dim, hidden_size=128, num_layers=1, dropout=0.2,
                 bidirectional=False, d_proj=32, K=10):
        super().__init__(input_dim, hidden_size, num_layers, dropout, bidirectional)
        self.spatial = SparseSpatialAttention(d_model=hidden_size, d_proj=d_proj, K=K)
        # Override head: 160 → 64 → 1
        head_dim = hidden_size // 2
        self.head = nn.Sequential(
            nn.Linear(hidden_size + d_proj, head_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(head_dim, 1),
        )

    def forward(self, x):
        out, _ = self.gru(x)
        h = out[:, -1, :]            # [N, hidden]
        c = self.spatial(h)          # [N, d_proj]
        hc = torch.cat([h, c], dim=-1)  # [N, hidden + d_proj]
        return self.head(hc).squeeze(-1)


class GRURankerSpatialGated(GRURanker):
    """S2: gated fusion. gate = σ(Linear(160→1))."""
    def __init__(self, input_dim, hidden_size=128, num_layers=1, dropout=0.2,
                 bidirectional=False, d_proj=32, K=10):
        super().__init__(input_dim, hidden_size, num_layers, dropout, bidirectional)
        self.spatial = SparseSpatialAttention(d_model=hidden_size, d_proj=d_proj, K=K)
        self.proj_c = nn.Linear(d_proj, hidden_size)  # 32 → 128
        self.gate = nn.Sequential(
            nn.Linear(hidden_size + d_proj, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        out, _ = self.gru(x)
        h = out[:, -1, :]            # [N, hidden]
        c = self.spatial(h)          # [N, d_proj]
        c_proj = self.proj_c(c)      # [N, hidden]
        g = self.gate(torch.cat([h, c], dim=-1))  # [N, 1]
        h_fused = g * h + (1 - g) * c_proj
        return self.head(h_fused).squeeze(-1)


class GRURankerSpatialRes(GRURanker):
    """S3: residual h + proj(context). proj = Linear(32→128)."""
    def __init__(self, input_dim, hidden_size=128, num_layers=1, dropout=0.2,
                 bidirectional=False, d_proj=32, K=10):
        super().__init__(input_dim, hidden_size, num_layers, dropout, bidirectional)
        self.spatial = SparseSpatialAttention(d_model=hidden_size, d_proj=d_proj, K=K)
        self.proj_c = nn.Linear(d_proj, hidden_size)  # 32 → 128

    def forward(self, x):
        out, _ = self.gru(x)
        h = out[:, -1, :]            # [N, hidden]
        c = self.spatial(h)          # [N, d_proj]
        h_fused = h + self.proj_c(c)  # [N, hidden]
        return self.head(h_fused).squeeze(-1)
