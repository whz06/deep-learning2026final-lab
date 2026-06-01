"""v6/models/spatial_attn.py — SparseSpatialAttention with d=32 bottleneck.

Corrected per SPATIAL_ATTENTION_PLAN.md Phase 1 design:
- Q/K/V project to d_proj=32 (bottleneck, not full 128-dim)
- Self-exclude via sim[i,i] = -inf before topk (cleaner than topk+remove)
- Returns context [N, d_proj] for fusion module to handle
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class SparseSpatialAttention(nn.Module):
    def __init__(self, d_model=128, d_proj=32, K=10):
        super().__init__()
        self.query = nn.Linear(d_model, d_proj, bias=False)
        self.key   = nn.Linear(d_model, d_proj, bias=False)
        self.value = nn.Linear(d_model, d_proj, bias=False)
        self.K = K
        self.scale = d_proj ** 0.5

    def forward(self, h):
        N = h.size(0)
        q = self.query(h)  # [N, d_proj]
        k = self.key(h)    # [N, d_proj]
        v = self.value(h)  # [N, d_proj]

        sim = q @ k.T / self.scale  # [N, N]
        sim.fill_diagonal_(-float('inf'))  # self-exclude

        K_eff = min(self.K, N - 1)
        topk_sim, topk_idx = sim.topk(K_eff, dim=-1)  # [N, K]
        attn = F.softmax(topk_sim, dim=-1)

        context = (attn.unsqueeze(1) @ v[topk_idx]).squeeze(1)  # [N, d_proj]
        return context
