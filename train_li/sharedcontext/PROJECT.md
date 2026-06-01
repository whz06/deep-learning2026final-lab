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
│   └── market/000300.SH.csv   # CSI300 index daily returns (for csi5d, csi10d, etc.)
├── processed/
│   ├── all_data.parquet       # Main feature source (328 MB, 2019–2026), output of shared/preprocess.py
│   └── v2_windows/            # Pre-built .pt window files for training (train/ + val/ by year)
├── v1/                        # Initial GRU baseline sweep
│   └── ...
├── v2/                        # Main model development (GRU, MLP, Transformer)
│   ├── config.py              # WINDOW_SIZE=60, INPUT_DIM=22, sweep grids, model configs
│   ├── train.py               # Training loop + sweep CLI (ListMLE loss, RankIC metric)
│   ├── dataset.py             # DailyStockDataset: loads .pt files, auto-truncates features[:, -W:, :]
│   ├── build_windows.py       # Builds .pt files: OHLCV + tech + cross-sectional → [N,60,22]
│   ├── infer.py               # Single-model inference (buy/sell list output)
│   │
│   ├── models/
│   │   ├── gru.py             # GRURanker: [B,T,22] → GRU → last hidden → MLP → scalar
│   │   ├── mlp.py             # MLPRanker: [B,T,22] → flatten → FC stack → scalar
│   │   └── transformer.py     # TransformerRanker: temporal + spatial attention
│   │
│   └── checkpoints/
│       ├── gru_gru_hidden_size=128_num_layers=1_dropout=0.2_lr=0.0003.pt   ← BEST (IC=0.1029)
│       ├── tf_tf_d_model=96_n_heads=4_n_temporal_layers=2_n_spatial_layers=1_dropout=0.1_lr=0.0003.pt
│       ├── mlp_mlp_hidden_dim=1024_n_layers=4_dropout=0.3_lr=0.0005.pt
│       └── gru_seed{42,123,456,789,1024}.pt   # Seed ensemble checkpoints
│
├── v3/                        # Strategy backtesting & analysis
│   ├── shared_precompute.py   # Precomputes daily GRU scores + CSI300 signals → benchmark_data.parquet
│   ├── results/
│   │   ├── benchmark_data.parquet   # 74-day signal data (Feb-May 2026), 600 sampled stocks
│   │   ├── strategy_b.json          # Best: momentum stop-loss (+6.85%, sharpe=0.93)
│   │   ├── strategy_a.json          # High-vol risk-off (+5.58%)
│   │   ├── strategy_dispersion.json # Score dispersion defense
│   │   └── diagnose_variance_drag.py # Beta amplification analysis
│   └── ...
│
├── v4/                        # Multi-window ensemble feasibility
│   ├── feasibility.py         # L1: momentum alpha proxy IC, L2: GRU W60 vs W90
│   ├── train_t30.py           # Train T=30 GRU (reuses existing .pt files, auto-truncates)
│   ├── eval_t30.py            # Compare T=30 vs T=60 on test period
│   ├── checkpoints/           # T=30 checkpoint goes here
│   └── results/
│       ├── feasibility.json
│       └── eval_t30.json      # Post-training evaluation output
│
├── trade/                     # Competition daily inference
│   ├── infer.py               # Self-contained: features + GRU model + Strategy B + buy/sell output
│   ├── buy_list.txt           # Next-day buy codes (top-N)
│   ├── sell_list.txt          # Next-day sell codes (bottom-K)
│   └── decision.log           # Daily position decisions (append-only)
│
└── sharedcontext/              # Project documentation (this folder)
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

| Model | Config | Val Rank IC |
|-------|--------|-------------|
| **GRU** | H=128, L=1, D=0.2, lr=3e-4, T=60 | **0.1029** |
| TF | d_model=96, n_heads=4, n_temporal=2, n_spatial=1, D=0.1, lr=3e-4 | 0.1009 |
| MLP | hidden=1024, n_layers=4, D=0.3, lr=5e-4 | 0.0843 |
| Fusion (val) | 0.6×GRU + 0.4×TF | 0.1057 |

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
