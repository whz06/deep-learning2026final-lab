# Experiments Log

Chronological record of all experiments, their results, and key takeaways.
Serves as reference material for the final report.

---

## V1 — GRU Baseline Sweep

**What**: Swept T×H×L×D×lr on GRU with 10 raw features only.

**Result**: Best `T=60, H=128, L=2, D=0.1, lr=5e-4`, val RankIC=0.1042.

---

## V2 — Multi-Model Sweep & Feature Expansion

### Feature Expansion
Extended from 10 raw → 22 dims (10 raw + 8 tech + 4 cross-sectional).

### Model Sweeps
| Model | Best Config | Val IC |
|-------|------------|--------|
| GRU | H=128, L=1, D=0.2, lr=3e-4 | 0.1029 |
| TF | d=96, heads=4, nt=2, ns=1, D=0.1, lr=3e-4 | 0.1009 |
| MLP | hidden=1024, n=4, D=0.3, lr=5e-4 | 0.0843 |

**Key finding**: L=1 beats L=2 on 22-dim features (less overfitting).

### Walk-Forward on 2026
10×10d segments, baseline +1.48% vs CSI300 +0.45%.

### Heterogeneous Ensemble (GRU+TF)
- Val fusion IC: 0.1057 (vs GRU 0.1029) — 60% GRU + 40% TF
- OOS underperformance: -0.78% — fusion degrades on test

### Seed Ensemble (GRU×5)
Minor variance reduction only. Score rank correlation between seeds: ~0.95 — pseudo-diversity.

---

## V3 — Strategy Backtesting

### Precompute Pipeline
`shared_precompute.py`: generates daily GRU scores + CSI300 signals → `v3/results/benchmark_data.parquet` (74 days, 600 stocks).

### Strategy C / A+C (Vol-Filtering)
- 162 + 81 configs tested
- **All underperformed baseline**
- Root cause: low-vol stocks don't protect in bear markets; vol-filtering eats alpha

### Strategy A (High-Vol Risk-Off)
- csi20vol > P80 → position = 80%
- Return: +5.58%, Sharpe: 0.79 (vs baseline +2.34%)
- Works but less robust than Strategy B

### Strategy B (Momentum Stop-Loss) — BEST
- csi5d < -1.0% → position = 80%
- Return: **+6.85%**, Sharpe: **0.93**
- 10-window test: 6/10 wins, 0 losses, mean improvement +0.64%/window
- Stability: first half +3.81%, second half +0.50% — robust across regimes
- Catches 8/10 worst days

### Variance Drag Diagnosis
- Model beta = 1.41 (amplifies market moves)
- 3 worst days eat 60 days of profit
- Strategy B closes 116% of the gap to CSI300

### Score Dispersion Defense
- Signal: score_std, score_skew, top20_score_std, n_above_z2
- P95 trigger catches 3 crash days Strategy B misses
- Combined: +7.67%, Sharpe: 1.05, 9/10 crash days caught

### Dispersion Deep-Dive — FAILED
- score_std is non-stationary (Feb-Apr avg 0.41, May avg 0.45+)
- High autocorrelation (lag-1 r=0.81)
- All 4 P95 triggers concentrated in last 10 days
- 10-window test: 0 wins / 3 losses / 7 ties
- **Conclusion**: dispersion defense NOT robust for competition

### Continuous Sizing — FAILED
- All linear/exponential/vol-adaptive mappings worse than binary threshold
- Best continuous: +5.56% vs binary +6.85%
- Binary "cliff" is a feature, not a bug

---

## V4 — Multi-Window Ensemble Feasibility

### Feasibility Check (feasibility.py)

**Layer 1 — Model-Free Momentum Alpha Proxy**:
| Window | Mean Rank IC | IC > 0 |
|--------|-------------|--------|
| mom_5d | -0.0036 | 50% |
| mom_20d | -0.0083 | 47% |
| mom_60d | -0.0144 | 43% |

Momentum alone has no predictive power in test period.

**IC Correlations**:
| Pair | Correlation | Interpretation |
|------|------------|----------------|
| 5d vs 20d | 0.523 | Moderate — related |
| 5d vs 60d | **0.138** | **Nearly independent — genuine diversity** |
| 20d vs 60d | 0.591 | Moderate-high |

**Layer 2 — GRU W60 vs W90**:
| Metric | Value |
|--------|-------|
| GRU W60 mean IC | +0.0423 (64% positive) |
| GRU W90 mean IC | +0.0430 (65% positive) |
| IC correlation W60-W90 | **0.997** |
| Score rank correlation | **0.9876 ± 0.0037** |

**Key Findings**:
- T=90 is **dead**: GRU on 90 days produces near-identical ranking to 60 days. Training a separate T=90 model won't add diversity.
- T=30 is the **only hope**: Layer 1 shows 5d vs 60d correlation = 0.138 — short and long horizons see fundamentally different information.
- Decision: train T=30 only, skip T=90.

### Implementation (train_t30.py + eval_t30.py)
- `train_t30.py`: Trains T=30 GRU with same architecture (H=128, L=1, D=0.2, lr=3e-4). Reuses existing v2_windows .pt files — dataset auto-truncates to W=30.
- `eval_t30.py`: Loads both T=30 and T=60 checkpoints, generates scores on test period (Feb-May 2026), computes score rank correlation.
- Decision rule: rank corr > 0.9 → skip ensemble; < 0.7 → equal-weight fusion; 0.7-0.9 → borderline.

### T=30 vs T=60 Evaluation Results

| Metric | T=30 | T=60 | Notes |
|--------|------|------|-------|
| Mean Rank IC (test) | **+0.0363** | **+0.0424** | T=30 weaker by 14% |
| IC > 0 fraction | 61% | 64% | Both barely positive |
| IC correlation | — | — | **0.962** — near-perfect correlation |
| Score rank correlation | — | — | **0.845 ± 0.030** |

**Three reasons ensemble fails:**

1. **T=30 standalone IC is strictly worse.** Fusing a weak model into a stronger one dilutes the signal.
2. **IC correlation = 0.962.** The daily win/loss pattern of both models is nearly identical. There are no "T=30 day" vs "T=60 day" — no regime specialization.
3. **Score rank correlation = 0.845.** Below the 0.9 "identical" threshold, but above the 0.7 "genuinely diverse" threshold. The 15% ranking difference is noise, not complementary alpha.

Quantitative estimate: any weighted fusion w·S30 + (1-w)·S60 yields lower IC than T=60 alone (w=0). The optimal weight is w=0.

**Conclusion: Multi-window ensemble (方向 2) is abandoned. T=60 + Strategy B remains the base model.**

---

## Trade/ — Competition Infrastructure

### infer.py — Self-Contained Daily Inference
- Inlines GRU model definition (27 lines) — no dependency on v2/
- Feature computation identical to training (add_tech + cross-sectional ranks)
- Strategy B auto-applied: csi5d < -1.0% → 80% position

### Output Files
| File | Description |
|------|-------------|
| `buy_list.txt` | Top-N stock codes for next-day buy |
| `sell_list.txt` | Bottom-K stock codes for next-day sell |
| `decision.log` | Append-only: date, position%, csi5d, trigger, buy_n, sell_k |

### Usage
```powershell
& D:\Software\miniconda3\envs\dl_lab1\python.exe D:\Workspace\DL_HW\LAB5\trade\infer.py --date 20260529
```

### Verified
- 20260529 (bullish): CSI5d=+0.98% → FULL position, buy=20 sell=5 ✓
- 20260323 (bearish): CSI5d=-5.51% → 80% position, buy=16 sell=4 ✓
- Buy list matches v2/buy_list.txt exactly ✓
