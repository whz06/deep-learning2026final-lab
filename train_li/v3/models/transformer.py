"""
v2/models/transformer.py — Transformer with temporal + spatial self-attention.

Each block:
  1. Temporal Self-Attn (per-stock, along T axis)
  2. Learnable temporal pooling → [N, d]
  3. Spatial Self-Attn (cross-stock, along N axis)
  4. Broadcast + residual back to [N, T, d]
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class AttentionPool(nn.Module):
    """Learnable attention pooling along T dimension: [N,T,d] → [N,d]."""
    def __init__(self, d_model: int):
        super().__init__()
        self.w = nn.Linear(d_model, 1)

    def forward(self, x):
        # x: [N, T, d]
        alpha = self.w(x).softmax(dim=1)   # [N, T, 1]
        return (alpha * x).sum(dim=1)       # [N, d]


class SpatialAttentionLayer(nn.Module):
    """Self-attention across stocks (N axis). Input: [N, d]."""
    def __init__(self, d_model: int, n_heads: int, dropout: float):
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout,
                                          batch_first=True)
        self.norm = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 2, d_model),
        )
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # x: [N, d] → reshape to [1, N, d] for MHA (treat stocks as sequence)
        h = x.unsqueeze(0)                              # [1, N, d]
        attn_out, _ = self.attn(h, h, h)                # [1, N, d]
        x = self.norm(x + self.dropout(attn_out.squeeze(0)))   # [N, d]
        x = self.norm2(x + self.dropout(self.ffn(x)))   # [N, d]
        return x


class SpatioTemporalBlock(nn.Module):
    """One block: temporal attn → pool → spatial attn → broadcast."""
    def __init__(self, d_model: int, n_heads: int, dropout: float):
        super().__init__()
        self.temporal_attn = nn.MultiheadAttention(d_model, n_heads,
                                                    dropout=dropout, batch_first=True)
        self.norm_t = nn.LayerNorm(d_model)
        self.pool = AttentionPool(d_model)
        self.spatial = SpatialAttentionLayer(d_model, n_heads, dropout)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # x: [N, T, d]
        # 1. Temporal self-attention (within each stock's T)
        t_out, _ = self.temporal_attn(x, x, x)          # [N, T, d]
        x = self.norm_t(x + self.dropout(t_out))         # [N, T, d]

        # 2. Pool to [N, d]
        h_pool = self.pool(x)                            # [N, d]

        # 3. Spatial attention across stocks
        h_spatial = self.spatial(h_pool)                 # [N, d]

        # 4. Broadcast back: [N, d] → [N, 1, d] → [N, T, d]
        x = x + h_spatial.unsqueeze(1).expand_as(x)      # [N, T, d]
        return x


class TransformerRanker(nn.Module):
    def __init__(self, input_dim: int, d_model: int = 96, n_heads: int = 4,
                 n_temporal_layers: int = 2, n_spatial_layers: int = 1,
                 dropout: float = 0.1):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, d_model)
        self.pos_embed = nn.Parameter(torch.randn(1, 60, d_model) * 0.02)

        blocks = []
        for i in range(max(n_temporal_layers, n_spatial_layers)):
            use_spatial = (i < n_spatial_layers)
            use_temporal = (i < n_temporal_layers)
            if use_temporal and use_spatial:
                blocks.append(SpatioTemporalBlock(d_model, n_heads, dropout))
            elif use_temporal:
                blocks.append(TemporalOnlyBlock(d_model, n_heads, dropout))
            else:
                blocks.append(SpatialOnlyBlock(d_model, n_heads, dropout))

        self.blocks = nn.ModuleList(blocks)
        self.head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, 1),
        )

    def forward(self, x):
        # x: [N, T, F]
        x = self.input_proj(x)                           # [N, T, d]
        x = x + self.pos_embed[:, :x.shape[1], :]        # [N, T, d]

        for block in self.blocks:
            x = block(x)                                 # [N, T, d]

        # Final pooling: mean over T
        h = x.mean(dim=1)                                # [N, d]
        return self.head(h).squeeze(-1)                   # [N]


class TemporalOnlyBlock(nn.Module):
    """Pure temporal attention (used when n_temporal > n_spatial)."""
    def __init__(self, d_model: int, n_heads: int, dropout: float):
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.norm = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(nn.Linear(d_model, d_model*2), nn.GELU(),
                                  nn.Dropout(dropout), nn.Linear(d_model*2, d_model))
        self.norm2 = nn.LayerNorm(d_model)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        out, _ = self.attn(x, x, x)
        x = self.norm(x + self.drop(out))
        x = self.norm2(x + self.drop(self.ffn(x)))
        return x


class SpatialOnlyBlock(nn.Module):
    """Spatial attention after pooling (used when n_spatial > n_temporal)."""
    def __init__(self, d_model: int, n_heads: int, dropout: float):
        super().__init__()
        self.pool = AttentionPool(d_model)
        self.spatial = SpatialAttentionLayer(d_model, n_heads, dropout)
        self.norm = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(nn.Linear(d_model, d_model*2), nn.GELU(),
                                  nn.Dropout(dropout), nn.Linear(d_model*2, d_model))
        self.norm2 = nn.LayerNorm(d_model)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        # x: [N, T, d]
        h = self.pool(x)                                 # [N, d]
        h = self.spatial(h)                              # [N, d]
        h = self.norm2(h + self.drop(self.ffn(self.norm(h))))
        x = x + h.unsqueeze(1).expand_as(x)              # broadcast back
        return x
