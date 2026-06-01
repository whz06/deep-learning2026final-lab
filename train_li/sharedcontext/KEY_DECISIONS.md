# Key Design Decisions & Rationale

## Why GRU, not Transformer

GRU val IC = 0.1029 vs TF = 0.1009. GRU simpler, faster, handles variable sequence lengths natively. TF has hardcoded `pos_embed(1, 60, d_model)` and would need modification for different windows. Also, heterogeneous ensemble (GRU+TF) degraded OOS — TF adds noise more often than signal.

## Why L=1 for GRU (not L=2)

V1 used L=2 on 10 features, but with 22 features (v2 expansion), L=1 outperforms L=2 (0.1029 vs lower). More layers cause overfitting on the richer feature set.

## Why ListMLE Loss

Listwise ranking loss, directly optimizes for correct ordering — matches the use case (buy top-N, sell bottom-K). Pairwise (LambdaRank) and pointwise (MSE) were not tested in this project; ListMLE came from v1 precedent and works well.

## Why Strategy B (Momentum Stop-Loss) Works

GRU's OOS IC is only 0.042 — the model's stock ranking edge is small. Strategy B's +6.85% return doesn't come from better stock picking; it comes from **avoiding market crashes**. When csi5d < -1.0%, reducing position to 80% prevents catastrophic losses.

This is the only robust strategy because:
- It's regime-agnostic (works in both bull and bear halves)
- It's binary (avoids overfitting noise with continuous sizing)
- The signal (CSI300 5-day return) is a direct market observation, not a model-derived metric

## Why Dispersion Defense Failed

score_std is non-stationary (concentrated in May 2026). Its P95 triggers look impressive in full-period backtest (+7.67%) but the signal is entirely concentrated in the last 10 days — a period of unusual market stress. In a 10-window test, it produces 0 wins, 3 losses, 7 ties. For a 10-day competition, it's useless.

## Why Continuous Sizing Failed

Moving from binary (100%/80%) to continuous (e.g., 100%→80% ramp based on csi5d severity) uniformly reduces returns. The binary "cliff" creates a strong behavioral signal: "either the market is normal, or it's in trouble." Softening this boundary reduces the protective effect without gaining anything.

## Why Multi-Window Ensemble (T=90, T=30) Was Abandoned

## Architecture for Competition Inference

`trade/infer.py` is self-contained:
- Inlines the GRU model definition (27 lines) rather than depending on v2/
- Copies feature computation logic (add_tech + cross-sectional ranks)
- Always runs Strategy B by default

This minimizes dependency chains — if v2/ changes, trade/ doesn't break.

## Why No Test Set for Validation

The training pipeline uses Train (2019-2024) → Val (2025). The 2026 period is the actual competition period. We don't have a separate "test set" before competition — the v3 backtests on 2026 Feb-May are "out-of-sample" with respect to model training, but the ground truth returns are known. The 10-day competition (June 1-12) is the true test, and its data doesn't exist yet.

## Summary of Failed Approaches

| Approach | Why It Failed |
|----------|--------------|
| C/A+C vol-filtering | Low-vol ≠ crash protection; filtering eats GRU alpha |
| GRU+TF fusion | TF adds noise most days; fusion IC gain is 0.0028 (within noise) |
| Dispersion defense | score_std non-stationary, signal concentrated in tail events |
| Continuous sizing | Binary threshold is a feature, softening reduces protection |
| Seed ensemble (GRU×5) | 0.95+ rank correlation between seeds = pseudo-diversity |
| T=90 multi-window | 0.9876 rank correlation with T=60 = complete redundancy |
| T=30 multi-window | T=30 IC 0.0363 < T=60 IC 0.0424; ensemble IC corr 0.962 = no complementarity |
