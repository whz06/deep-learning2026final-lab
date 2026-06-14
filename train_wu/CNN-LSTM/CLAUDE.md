# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project overview

GN-CNN-LSTM: a stock price prediction system that uses deep learning (LSTM / CNN / CNN-LSTM) to forecast next-day closing prices. Predictions feed into an N-K rotation backtesting engine that simulates a portfolio strategy (hold N stocks, rotate up to K per day).

## Commands

```bash
# Train (--model: lstm/cnn/cnn_lstm, --scale: maxmin/sigmoid/all, --seq_lens: comma-separated)
python train.py --model lstm --scale all --seq_lens 5,10,20

# Train with device and year filter
python train.py --model cnn_lstm --scale sigmoid --seq_lens 10,20 --device auto --year 2024

# Generate recent buy recommendations from a trained model
python predict.py --model_path "result\cnn_lstm_L5.pt" --last_days 3 --top_k 10

# Standalone backtest on saved test signals
python backtest.py --signals_csv "result\lstm_L20_test_signals.csv.gz" --n 10 --k 3

# Helper scripts (in scripts/)
python scripts/augment_data.py        # Data augmentation
python scripts/test_load.py           # Test data loading
python scripts/test_build.py          # Test window construction
```

## Architecture

```
src/stock_predictor/
  config.py    — global constants, TrainConfig dataclass (batch_size, lr, patience, lstm_hidden, cnn_filters…)
  data.py      — CSV loading, sliding-window construction (numpy vectorized), inverse normalization (maxmin/sigmoid), train/val/test split
  models.py    — LSTMRegressor (2-layer LSTM), CNNRegressor (3×Conv1d), CNNLSTMRegressor (Conv1d→MaxPool→LSTM), build_model() factory, pick_device()
  trainer.py   — training loop with per-batch early stopping, predict_arrays(), save/load checkpoint
  metrics.py   — regression metrics (MSE/MAE/RMSE/R2), IC/IR/RankIC/RankIR, direction win rate; all computed per 12-day period then averaged
  strategy.py  — N-K rotation backtest (T+1, lot_size=100), top_picks_report markdown generator, save_backtest()
train.py        — CLI entry point: loops over scale × seq_len combos, trains, evaluates, backtests
predict.py      — CLI for inference: loads checkpoint, builds recent windows, outputs signals + top-K report
backtest.py     — CLI for standalone backtest on existing signals CSV
scripts/        — helper scripts (augment_data, test_load, test_build, append_jun8)
sim_data/       — simulated daily close data CSV files for prediction
result/         — model checkpoints (.pt), metrics JSON, test signals (.csv.gz), backtest results, IC plots, experiment reports
```

## Data flow

1. Input CSVs: `maxmin_scale_daily_close.csv` (cols: ts_code, trade_date, close_raw, min_ref, max_ref, close_scaled, train) and `sigmoid_scale_daily_close.csv` (cols: ts_code, trade_date, close, scaled_close)
2. `load_scale_frame()` reads one CSV, normalizes column names, assigns train/test split (by stock, first 80% of rows = train)
3. `build_splits()` creates overlapping sliding windows of length `seq_len` (vectorized via `np.lib.stride_tricks.sliding_window_view`), further splitting train into train/val (last 10% of train range of each stock)
4. Features = `scaled_value` (1-dim), target = next day's `scaled_value`
5. After prediction, `predict_arrays()` inverts predictions back to raw prices via `inverse_maxmin()` or `inverse_sigmoid_formula()` and computes `pred_ret = pred_raw / cur_close - 1`
6. Backtest uses `pred_ret` to rank stocks, holds top N, rotates up to K per day with T+1 execution at lot_size=100

## Key design decisions

- **Two-scale merging**: model trains on one scale at a time (maxmin or sigmoid), not both simultaneously. `--scale all` runs two separate training passes
- **Stock-level split**: the `is_train` column guarantees all windows for a given stock are in exactly one split (no leakage across train/val/test for the same stock). Split is determined per-stock by row index, not globally shuffled
- **Early stopping**: tracked per batch (not per epoch), triggered by validation MSE stagnation over `PATIENCE` consecutive log steps. Validation happens every `LOG_EVERY_BATCHES = 500` batches
- **Period metrics**: all evaluation metrics (IC, IR, direction win rate) are computed by grouping consecutive trading days into 12-day periods, computing metrics per period, then averaging — not computed globally across all data
- **Sigmoid inverse**: uses formula `x = START + N - (N/A) * (ln(2/score - 1) - B)` where `START=0, END=150, N=150, A=ln(40000), B=ln(0.005)`. Inputs are stored as `score - 1` in the CSV so the formula first adds 1 back
- **PriceWindowDataset**: PyTorch Dataset wrapping WindowArrays; each sample returns (x, y_scaled, y_raw, cur_close, next_close, min_ref, max_ref) — the extra metadata is carried through for evaluation but only `x` and `y_scaled` are used in training loss
- **Missing value fill**: `fill_missing_with_mean()` in metrics.py fills NaN values with the column mean before computing evaluation metrics (affects sigmoid scale inference results)
- **Inference batch size**: `predict_arrays()` uses `batch_size=4096` for faster inference (compared to 256 for training)
- **N-K grid search**: backtest N and K parameters can be tuned — `result/nk_grid_search.csv` stores results across N/K combinations

## Backtest rules

- T+1 execution: signal on day D, trade executed at day D+1 close
- Lot size = 100 shares per trade (rounds down to nearest lot)
- Hold N stocks at all times, rotate at most K per day
- First day: buy top N stocks outright
- Subsequent days: sell from lowest-ranked held stocks, buy highest-ranked unheld stocks (up to K each)
- Equal-weight target: after sells, cash is divided equally among N positions for buys
- Metrics: annual return, max drawdown, Sharpe ratio (annualized, 252 trading days)
- Results saved to `result/backtest/` directory