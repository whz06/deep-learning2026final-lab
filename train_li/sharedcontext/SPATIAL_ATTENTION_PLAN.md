# Spatial Attention Plan (Archived — superseded by plan0601.md)

This was the original v5 plan for GRU + Sparse Spatial Attention, run BEFORE the cross-sectional feature bug was discovered. Abandoned because training on 18 effective features (4 cross features were all zero due to normalization bug) would produce a suboptimal model.

## Architecture

```
[N, T, 22] → GRU → h [N, 128] → SparseSpatialAttn → h_enhanced → Head → scores [N]
```

### SparseSpatialAttn
- Q/K/V projections: h → learnable linear layers → [N, d]
- Similarity: sim = q @ k^T / √d [N, N]
- Self-exclude: sim[i,i] = -inf
- Top-K sparsification: keep K highest → attn weights via softmax
- Context aggregation: weighted sum of neighbor value vectors
- Fusion with original h: concat | gated | residual

## Phase 1 — Architecture Validation (4 configs)

Fixed: T=60, H=128, L=1, D=0.2, lr=3e-4, N_sample=1024, d=32, K=10

| ID | Fusion | Head Input | Description |
|----|--------|-----------|-------------|
| S0 | none | 128 | Baseline GRU (N_sample=2048) |
| S1 | concat | 160 | [h ‖ context] → head |
| S2 | gated | 128 | gate * h + (1-gate) * proj(context) |
| S3 | residual | 128 | h + proj(context) |

Selection: val Rank IC > baseline (0.1029) → spatial attention works.

## Phase 2 — Best K (2-3 configs)

Take best fusion from Phase 1.

| ID | K | Purpose |
|----|---|---------|
| S4 | 5 | Tighter neighborhood |
| S5 | 20 | Broader neighborhood |

## Phase 3 — Optimize (2-3 configs)

- proj_dim: 16, 64
- N_sample: 2048 (full size)
- Temperature scaling

## Phase 4 — Test Period Validation

Eval on Feb-May 2026: compare IC, score rank correlation, Strategy B backtest.

## Why Abandoned

P0 bug found: cross-sectional features (rank(pct_chg), etc.) tiled to all timesteps then Z-scored → all become zero. Model trains on 18 effective features, not 22. Spatial attention finding neighbors based on incomplete hidden states → suboptimal. Correct order: fix features (Step 1) → train baseline → THEN add spatial attention (Step 3).

## Reference

- Model code was in v5/model.py (deleted)
- Training code was in v5/train.py (deleted)
- KNN proxy verification: v4/verify_spatial.py (+0.0049 IC gain, K=10, α=0.5)
