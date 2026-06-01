"""
Step 2: Build sliding windows, normalize per-window, save as .pt files per trading day.

Input:
  ../processed/all_data.parquet
Output:
  ../processed/windows/train/{year}/{date}.pt
  ../processed/windows/val/{year}/{date}.pt

Each .pt file:
  {"features": torch.Tensor [N, T, F], "labels": torch.Tensor [N], "ts_codes": list[str]}
"""
import os
import sys
import gc
import numpy as np
import pandas as pd
import torch
from collections import defaultdict

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PARQUET_PATH = os.path.join(ROOT, "processed", "all_data.parquet")
WINDOWS_DIR = os.path.join(ROOT, "processed", "windows")

WINDOW_SIZE = 60
FEATURE_COLS = ["open", "high", "low", "close", "vol", "amount", "pct_chg",
                "turnover_rate", "volume_ratio", "total_mv"]
LABEL_COL = "pct_chg"

TRAIN_START = "20190102"
TRAIN_END = "20241231"
VAL_START = "20250102"
VAL_END = "20251231"

FLUSH_INTERVAL_STOCKS = 500
MEMORY_LIMIT_BYTES = 2 * 1024**3  # 2 GB buffer before flushing


def normalize_window(window: np.ndarray) -> np.ndarray:
    mean = window.mean(axis=0, keepdims=True)
    std = window.std(axis=0, keepdims=True) + 1e-8
    return (window - mean) / std


def get_split(trade_date: str):
    if TRAIN_START <= trade_date <= TRAIN_END:
        return "train"
    elif VAL_START <= trade_date <= VAL_END:
        return "val"
    return None


def flush_dates(date_buffers, flush_keys):
    """Save specified date-keys to disk as .pt, remove from buffer, free memory."""
    saved = 0
    for key in flush_keys:
        if key not in date_buffers:
            continue
        buf = date_buffers.pop(key)
        n = len(buf["labels"])
        if n == 0:
            continue

        split, date = key
        out_dir = os.path.join(WINDOWS_DIR, split, date[:4])
        os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join(out_dir, f"{date}.pt")

        features = np.stack(buf["features"], axis=0)
        torch.save({
            "features": torch.from_numpy(features).float(),
            "labels": torch.tensor(buf["labels"], dtype=torch.float32),
            "ts_codes": buf["ts_codes"],
        }, out_path)
        saved += 1

    gc.collect()
    return saved


def estimate_buffer_mb(date_buffers):
    """Rough estimate of in-memory size of all buffers in MB."""
    total = 0
    for buf in date_buffers.values():
        n = len(buf["labels"])
        total += n * (WINDOW_SIZE * len(FEATURE_COLS) * 4 + 4 + 12)
    return total / (1024 * 1024)


def main():
    print(f"[build_windows] Loading {PARQUET_PATH} ...")
    df = pd.read_parquet(PARQUET_PATH)
    df["trade_date"] = df["trade_date"].astype(str)
    print(f"[build_windows] Loaded {len(df)} rows, {df['ts_code'].nunique()} stocks")

    stocks = sorted(df["ts_code"].unique())
    total_stocks = len(stocks)

    date_buffers = defaultdict(lambda: {"features": [], "labels": [], "ts_codes": []})
    total_saved = 0

    for s_idx, ts_code in enumerate(stocks):
        stock_df = df[df["ts_code"] == ts_code].sort_values("trade_date").reset_index(drop=True)
        vals = stock_df[FEATURE_COLS].values.astype(np.float32)

        if len(stock_df) < WINDOW_SIZE + 1:
            continue

        for i in range(WINDOW_SIZE - 1, len(stock_df) - 1):
            window = vals[i - WINDOW_SIZE + 1: i + 1]
            label = stock_df[LABEL_COL].iloc[i + 1]

            if np.isnan(window).any() or np.isnan(label):
                continue

            trade_date = stock_df["trade_date"].iloc[i]
            split = get_split(trade_date)
            if split is None:
                continue

            window = normalize_window(window)
            date_buffers[(split, trade_date)]["features"].append(window)
            date_buffers[(split, trade_date)]["labels"].append(float(label))
            date_buffers[(split, trade_date)]["ts_codes"].append(ts_code)

        # Periodic flush
        if (s_idx + 1) % FLUSH_INTERVAL_STOCKS == 0:
            mem_mb = estimate_buffer_mb(date_buffers)
            print(f"  stock {s_idx + 1}/{total_stocks} | buffer: {mem_mb:.0f} MB | "
                  f"dates buffered: {len(date_buffers)}")

            mem_limit_mb = MEMORY_LIMIT_BYTES / (1024 * 1024)
            if mem_mb > mem_limit_mb:
                all_keys = sorted(date_buffers.keys())
                flush_cutoff = len(all_keys) // 3
                keys_to_flush = all_keys[:flush_cutoff]
                n = flush_dates(date_buffers, keys_to_flush)
                total_saved += n
                print(f"  flushed {n} date files, {len(date_buffers)} dates remaining")

    # Final flush
    print(f"[build_windows] Final flush: {len(date_buffers)} dates remaining ...")
    all_keys = sorted(date_buffers.keys())
    n = flush_dates(date_buffers, all_keys)
    total_saved += n
    print(f"[build_windows] Done. Total date files saved: {total_saved}")

    # Print summary counts
    train_dates = len([k for k in all_keys if k[0] == "train"])
    val_dates = len([k for k in all_keys if k[0] == "val"])
    print(f"[build_windows] Train dates: {train_dates}, Val dates: {val_dates}")


if __name__ == "__main__":
    main()
