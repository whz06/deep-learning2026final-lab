"""v8/models/spatial_attn_v8.py — SparseSpatialAttention with industry gating."""
import torch
import torch.nn as nn
import torch.nn.functional as F


class SparseSpatialAttention(nn.Module):
    def __init__(self, d_model=128, d_proj=32, K=10):
        super().__init__()
        self.query = nn.Linear(d_model, d_proj, bias=False)
        self.key = nn.Linear(d_model, d_proj, bias=False)
        self.value = nn.Linear(d_model, d_proj, bias=False)
        self.K = K
        self.scale = d_proj ** 0.5

    def forward(self, h, industry_id=None, lambda_gate=0.1):
        N = h.size(0)
        q = self.query(h)  # [N, d_proj]
        k = self.key(h)
        v = self.value(h)

        sim = q @ k.T / self.scale  # [N, N]
        sim.fill_diagonal_(-float('inf'))

        # Industry gating: boost same-industry pairs
        if industry_id is not None and lambda_gate > 0:
            same_ind = (industry_id.unsqueeze(1) == industry_id.unsqueeze(0)).float()
            same_ind.fill_diagonal_(0.0)
            sim = sim + lambda_gate * same_ind

        K_eff = min(self.K, N - 1)
        topk_sim, topk_idx = sim.topk(K_eff, dim=-1)
        attn = F.softmax(topk_sim, dim=-1)
        context = (attn.unsqueeze(1) @ v[topk_idx]).squeeze(1)
        return context
