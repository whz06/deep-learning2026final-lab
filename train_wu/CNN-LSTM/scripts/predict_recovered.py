"""
Predict using recovered cnn_lstm_L20.pt + data with Jun 1-8 data.
Adds dummy Jun 9 rows so build_recent_windows uses Jun 8 as signal.
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
from stock_predictor.strategy import top_picks_report


def load_old_checkpoint(model_path: Path, device_name: str = "auto"):
    ckpt = torch.load(model_path, map_location="cpu", weights_only=False)
    cfg = ckpt["model_cfg"]
    seq_len = ckpt.get("seq_len", cfg.get("seq_len", 20))
    model_name = cfg.get("model_name", cfg.get("name", "cnn_lstm"))
    scale_name = "maxmin"  # old model was trained on maxmin
    device = pick_device(device_name)
    model = build_model(
        model_name=model_name, input_dim=1, seq_len=seq_len,
        hidden=cfg.get("lstm_hidden", 256),
        filters=cfg.get("cnn_filters", 64),
        dropout=cfg.get("dropout", 0.2),
    )
    # Remap old key names (fc -> head) to match current model code
    state_dict = ckpt["state_dict"]
    key_map = {"fc": "head"}
    remapped = {}
    for k, v in state_dict.items():
        new_k = k
        for old, new in key_map.items():
            new_k = new_k.replace(old, new)
        remapped[new_k] = v
    model.load_state_dict(remapped)
    model = model.to(device)
    model.eval()
    return model, scale_name, seq_len, device


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default="result/cnn_lstm_L20.pt")
    parser.add_argument("--top_k", type=int, default=10)
    parser.add_argument("--device", type=str, default="auto")
    args = parser.parse_args()

    model_path = Path(args.model_path)
    print(f"Loading model from {model_path}...", flush=True)
    model, scale_name, seq_len, device = load_old_checkpoint(model_path, args.device)
    print(f"  Model: CNN-LSTM, scale={scale_name}, seq_len={seq_len}", flush=True)

    print(f"Loading data ({scale_name})...", flush=True)
    df = load_scale_frame(scale_name)
    print(f"  {len(df)} rows, {df['ts_code'].nunique()} stocks", flush=True)
    dates = sorted(df["trade_date"].unique())
    print(f"  Latest dates: {dates[-10:]}", flush=True)

    # Add dummy Jun 9 rows so Jun 8 is signal, Jun 9 is target
    print("Adding dummy Jun 9 placeholder rows...", flush=True)
    dummy_rows = []
    for ts_code, g in df.groupby("ts_code"):
        last = g.iloc[-1]
        if last["trade_date"] == 20260608:
            dummy_rows.append({
                "ts_code": ts_code, "trade_date": 20260609,
                "scaled_value": last["scaled_value"],
                "close_raw": last["close_raw"],
                "min_ref": last["min_ref"],
                "max_ref": last["max_ref"],
                "is_train": 0,
            })
    dummy_df = pd.DataFrame(dummy_rows)
    df = pd.concat([df, dummy_df], ignore_index=True)
    df = df.sort_values(["ts_code", "trade_date"], kind="mergesort").reset_index(drop=True)
    print(f"  Added {len(dummy_rows)} dummy rows", flush=True)
    print(f"  Total: {len(df)} rows", flush=True)

    print(f"Building recent windows (last 1 day, seq_len={seq_len})...", flush=True)
    arrays = build_recent_windows(df, scale_name, seq_len, last_n_days=1)
    print(f"  {len(arrays.x)} windows built", flush=True)
    if len(arrays.x) > 0:
        sig_dates = sorted(set(int(d) for d in arrays.signal_date))
        tgt_dates = sorted(set(int(d) for d in arrays.target_date))
        print(f"  Signal dates: {sig_dates}", flush=True)
        print(f"  Target dates: {tgt_dates}", flush=True)
        n_stocks = len(set(arrays.ts_code))
        print(f"  Unique stocks: {n_stocks}", flush=True)

    if len(arrays.x) == 0:
        print("ERROR: No windows built!", flush=True)
        return

    print("Predicting...", flush=True)
    signals = predict_arrays(model, arrays, batch_size=4096, device=device, scale_name=scale_name)
    print(f"  {len(signals)} signals generated", flush=True)
    print(f"  pred_ret range: {signals['pred_ret'].min():.4f} ~ {signals['pred_ret'].max():.4f}", flush=True)

    prefix = model_path.stem
    signal_path = RESULT_DIR / f"{prefix}_recent_signals.csv.gz"
    report_path = RESULT_DIR / f"{prefix}_recent_top{args.top_k}.md"

    signals.to_csv(signal_path, index=False, encoding="utf-8")
    top_picks_report(signals, top_k=int(args.top_k), last_n_days=1, out_path=report_path)

    print(f"\n✅ Results saved:")
    print(f"   Signals: {signal_path}")
    print(f"   Report:  {report_path}")


if __name__ == "__main__":
    main()
