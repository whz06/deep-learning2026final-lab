# Project Overview — A-Share Stock Ranking (DL_HW LAB5)

## Goal
Deep learning pipeline for A-share stock ranking prediction (T+1 return), feeding a 10-day simulated trading competition (June 1–12, 2026) with daily buy/sell list output.

## Directory Structure

```
train_li/
├── shared/                    # Shared utilities (v1 era)
│   └── preprocess.py          # Merges data/daily/*.csv + data/metric/*.csv → processed/all_data.parquet
├── data/
│   ├── daily/                 # Daily stock CSV files (one per date, ~5000 stocks)
│   ├── metric/                # Daily financial metrics (pe/pb/circ_mv etc.)
│   ├── moneyflow/             # Daily money flow (19 cols: buy/sell by size class)
│   ├── market/000300.SH.csv   # CSI300 index daily returns
│   └── basic.csv              # Stock info (ts_code, name, industry, market, list_date)
├── processed/
│   ├── all_data.parquet       # Main feature source (2016–2026, ~4959 stocks × 2526 dates)
│   ├── v2_windows/            # Pre-built .pt window files (22-dim, v2 era)
│   ├── v5_windows/            # 26-dim windows (v5-v7, T+5→T+1)
│   └── v7_windows/            # 26-dim windows (v7, T+1 label, LABEL_HORIZON=1)
├── v1/                        # Initial GRU baseline sweep
├── v2/                        # Main model development (GRU, MLP, Transformer)
├── v3/                        # Strategy backtesting & analysis
├── v4/                        # Multi-window ensemble feasibility
├── v5/                        # Cross-sectional fix + preprocessing upgrade
├── v6/                        # Spatial attention (Step 3/4) + T+5 gap diagnosis
├── v7/                        # T+1 label training → best model (Spatial N=5 K=3)
├── v8/                        # Feature expansion (moneyflow+Tier1) + industry emb + loss comparison
│   ├── models/
│   ├── checkpoints/
│   └── results/
├── trade/                     # Competition daily inference
├── sharedcontext/              # Project documentation (this folder)
└── .gitignore
```

## Data Flow

```
data/daily/*.csv ─────┐
                       ├──→ shared/preprocess.py → processed/all_data.parquet
data/metric/*.csv ─────┘
                                      │
                                      ├──→ v2/build_windows.py → processed/v2_windows/{train,val}/{date}.pt
                                      │      Each .pt: {features:[N,60,22], labels:[N], ts_codes:[N]}
                                      │
                                      ├──→ v2/train.py → GRU/MLP/TF training → v2/checkpoints/
                                      │
                                      ├──→ v2/infer.py → buy_list.txt / sell_list.txt
                                      │
                                      └──→ v3/shared_precompute.py → benchmark_data.parquet
                                                                        │
                                                                        └──→ strategy backtesting
```

## Feature Engineering (22 dimensions)

| Group | Features | Count |
|-------|----------|-------|
| Raw | open, high, low, close, vol, amount, pct_chg, turnover_rate, volume_ratio, total_mv | 10 |
| Technical | macd, macd_signal, rsi, bb_width, bb_pct, mom_5, mom_20, vol_20 | 8 |
| Cross-sectional | rank(pct_chg), rank(amount), rank(turnover_rate), rel_beta (stock_pct_chg - idx_pct_chg) | 4 |

Z-score normalized per-window (cross-sectional mean/std across all stocks on that date).

## Training Config

| Parameter | Value |
|-----------|-------|
| Data split | Train: 2019–2024 (~1456 days), Val: 2025 (243 days) |
| Loss | ListMLE (listwise ranking) |
| Metric | Spearman Rank IC |
| Optimizer | AdamW, weight_decay=1e-5 |
| Scheduler | ReduceLROnPlateau (factor=0.5, patience=5, mode="max") |
| Early stop | 10 epochs no improvement |
| Grad clip | 1.0 |
| Batch size | 1 (one trading day per iteration), subsample 2048 stocks/day |

## Best Models

### Training Results (Val IC on 2025)

| Version | Model | Config | Val IC |
|--------|-------|--------|:---:|
| v2 | GRU | H=128 L=1 D=0.2 lr=3e-4 T=60 (22-dim) | 0.1029 |
| v2 | TF | d=96 heads=4 temporal=2 spatial=1 D=0.1 | 0.1009 |
| v5.2 | GRU (T+5) | H=128 L=1 D=0.2 (26-dim, log1p+winsor) | **0.1114** |
| v6 | Spatial (T+5) | d=32 K=5 concat (26-dim) | **0.1134** |
| v7 | GRU (T+1) | H=128 L=1 D=0.2 (26-dim) | 0.1023 |
| v7 | **Spatial (T+1)** | d=32 K=5 concat (26-dim) | **0.1062** ← current best |
| v8 | Spatial (T+1) | d=32 K=5 concat (31-dim + ind emb) | 0.1037 |

### Test Period Results (Jan 5 – May 29, 2026, 94 days, net of costs)

| Version | Model | N | K | Net Cum | Excess vs CSI300 | Sharpe |
|---------|-------|:---:|:---:|:---:|:---:|:---:|
| v2 | GRU | 20 | 5 | — | +6.85% (Feb-May) | 0.93 |
| v7 | **Spatial** | **5** | **3** | **+65.83%** | **+59.61%** | **5.59** |

### Trading Strategy (Production)

| Parameter | Value | Source |
|-----------|-------|--------|
| Model | v7 Spatial T+1 (26-dim, d=32, K=5) | best val+test IC |
| N_hold | 5 | N/K sweep optimal |
| K_sell | 3 | N/K sweep optimal |
| Loss | ListMLE | only viable option (see EXP_V8.md §3) |
| Position reduction | CSI5d < **-1.0%** → 80% | Strategy B, verified optimal (see EXP_V8.md §6) |
| Inference script | `trade/infer.py` | supports v2/v5/v6/v7/v8 autodetect, default v7 Spatial |


## Hardware & Constraints

- **GPU**: RTX 4060 Laptop 8GB VRAM
- **RAM**: 32GB
- **OS**: Windows, Python in WSL edits, PowerShell executes on Windows conda `dl_lab1`
- **CUDA memory**: `max_split_size_mb:128` (WDDM mode, no expandable_segments)
- **Trading rules**: no shorting, ≥80% position (max 20% cash), T+1 rotation
- **Data rules**: no future info leakage, time-series split (no random shuffle)

## Test Period

Feb 3 – May 29, 2026 (74 trading days), 600 sampled stocks.
Model OOS Rank IC: 0.0423 (Spearman) on test period. IC > 0 on 64% of days.
