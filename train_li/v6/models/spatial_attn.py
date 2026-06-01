"""v6/models/spatial_attn.py — SparseSpatialAttention (Step 3).

O(NK) KNN-sparsified cross-sectional attention.
Each stock attends to its K most similar neighbors (by learned Q/K similarity),
excluding self. Residual connection: h_out = h + context.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class SparseSpatialAttention(nn.Module):
    def __init__(self, d_model=128, K=10):
        super().__init__()
        self.query = nn.Linear(d_model, d_model, bias=False)
        self.key   = nn.Linear(d_model, d_model, bias=False)
        self.value = nn.Linear(d_model, d_model, bias=False)
        self.K = K
        self.scale = d_model ** 0.5

    def forward(self, h):
        N = h.size(0)
        q = self.query(h)   # [N, d]
        k = self.key(h)     # [N, d]
        v = self.value(h)   # [N, d]

        sim = q @ k.T / self.scale  # [N, N]

        # Top-K + exclude self
        topk_sim, topk_idx = sim.topk(min(self.K + 1, N), dim=-1)

        # Build mask to remove self (row index == col index)
        row_idx = torch.arange(N, device=h.device).unsqueeze(1)
        mask = topk_idx != row_idx

        # Gather K non-self neighbors per row
        gather_idx = mask.int().argsort(dim=-1, descending=True)[:, :self.K]
        topk_idx_k = topk_idx.gather(1, gather_idx)
        topk_sim_k = topk_sim.gather(1, gather_idx)

        attn = F.softmax(topk_sim_k, dim=-1)       # [N, K]
        context = (attn.unsqueeze(1) @ v[topk_idx_k]).squeeze(1)  # [N, d]

        return h + context
