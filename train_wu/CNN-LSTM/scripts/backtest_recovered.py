"""
Generate test signals from recovered model and run backtest with N=3.
Uses all recent test windows up to Jun 5 (signal) -> Jun 8 (target).
"""
from __future__ import annotations
import sys, argparse
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import torch
import numpy as np
import pandas as pd
from stock_predictor.config import RESULT_DIR
from stock_predictor.data import load_scale_frame, build_recent_windows
from stock_predictor.models import build_model, pick_device
from stock_predictor.trainer import predict_arrays
from stock_predictor.strategy import run_backtest, save_backtest


def load_old_checkpoint(model_path: Path, device_name: str = "auto"):
    ckpt = torch.load(model_path, map_location="cpu", weights_only=False)
    cfg = ckpt["model_cfg"]
    seq_len = ckpt.get("seq_len", cfg.get("seq_len", 20))
    model_name = cfg.get("model_name", cfg.get("name", "cnn_lstm"))
    scale_name = "maxmin"
    device = pick_device(device_name)
    model = build_model(
        model_name=model_name, input_dim=1, seq_len=seq_len,
        hidden=cfg.get("lstm_hidden", 256),
        filters=cfg.get("cnn_filters", 64),
        dropout=cfg.get("dropout", 0.2),
    )
    state_dict = ckpt["state_dict"]
    remapped = {}
    for k, v in state_dict.items():
        remapped[k.replace("fc", "head")] = v
    model.load_state_dict(remapped)
    model = model.to(device)
    model.eval()
    return model, scale_name, seq_len, device


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default="result/cnn_lstm_L20.pt")
    parser.add_argument("--last_days", type=int, default=100)
    parser.add_argument("--n", type=int, default=3)
    parser.add_argument("--k", type=int, default=1)
    parser.add_argument("--device", type=str, default="auto")
    args = parser.parse_args()

    model_path = Path(args.model_path)
    print(f"Loading model...", flush=True)
    model, scale_name, seq_len, device = load_old_checkpoint(model_path, args.device)
    print(f"  CNN-LSTM, scale={scale_name}, seq_len={seq_len}", flush=True)

    print(f"Loading data...", flush=True)
    df = load_scale_frame(scale_name)
    dates = sorted(df["trade_date"].unique())
    print(f"  {len(df)} rows, {df['ts_code'].nunique()} stocks", flush=True)
    print(f"  Date range: {dates[0]} ~ {dates[-1]}", flush=True)

    print(f"Building test windows (last {args.last_days} days)...", flush=True)
    arrays = build_recent_windows(df, scale_name, seq_len, last_n_days=args.last_days)
    print(f"  {len(arrays.x)} windows built", flush=True)
    if len(arrays.x) > 0:
        sig_dates = sorted(set(int(d) for d in arrays.signal_date))
        tgt_dates = sorted(set(int(d) for d in arrays.target_date))
        print(f"  Signal dates: {sig_dates[0]} ~ {sig_dates[-1]} ({len(sig_dates)} unique)", flush=True)
        print(f"  Target dates: {tgt_dates[0]} ~ {tgt_dates[-1]}", flush=True)

    if len(arrays.x) == 0:
        print("No windows built!", flush=True)
        return

    print(f"Predicting...", flush=True)
    signals = predict_arrays(model, arrays, batch_size=4096, device=device, scale_name=scale_name)

    # Filter to only signals where we have actual returns (target before Jun 9)
    signals = signals[np.isfinite(signals["real_ret"].to_numpy())].copy()
    print(f"  {len(signals)} signals with valid real_ret", flush=True)
    print(f"  pred_ret: {signals['pred_ret'].min():.4f} ~ {signals['pred_ret'].max():.4f}", flush=True)

    # Save full signals
    prefix = model_path.stem
    signal_path = RESULT_DIR / f"{prefix}_test_signals.csv.gz"
    signals.to_csv(signal_path, index=False, encoding="utf-8")
    print(f"  Signals saved: {signal_path}", flush=True)

    # Run backtest with N=3, K=1
    print(f"\nRunning backtest (N={args.n}, K={args.k})...", flush=True)
    daily, holdings, summary = run_backtest(signals, n=int(args.n), k=int(args.k))

    bt_saved = save_backtest(f"{prefix}_n{args.n}_k{args.k}", daily, holdings, summary)

    print(f"\n{'='*50}", flush=True)
    print(f"  回测结果 (N={args.n}, K={args.k})", flush=True)
    print(f"{'='*50}", flush=True)
    print(f"  年化收益率: {summary['annual_return']*100:.2f}%", flush=True)
    print(f"  最大回撤:   {summary['max_drawdown']*100:.2f}%", flush=True)
    print(f"  Sharpe比率:  {summary['sharpe']:.4f}", flush=True)
    print(f"  最终净值:   {summary['final_value']:.2f}", flush=True)
    print(f"  交易天数:   {summary['n_days']}", flush=True)
    print(f"{'='*50}", flush=True)

    # Show daily returns for the last few days
    print(f"\n最近交易日收益:", flush=True)
    last_days = daily.tail(10)
    for _, row in last_days.iterrows():
        print(f"  {int(row['signal_date'])} -> {int(row['target_date'])}: "
              f"收益={float(row['daily_ret'])*100:.2f}%, "
              f"持仓={int(row['n_hold'])}只, "
              f"净值={float(row['total_value']):.2f}", flush=True)

    print(f"\n回测文件:", flush=True)
    print(f"  Daily:    {bt_saved['daily']}", flush=True)
    print(f"  Holdings: {bt_saved['holdings']}", flush=True)
    print(f"  Summary:  {bt_saved['summary']}", flush=True)


if __name__ == "__main__":
    main()
