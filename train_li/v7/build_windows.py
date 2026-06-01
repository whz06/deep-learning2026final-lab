"""v5/build_windows.py — T+5 label + preprocessing + fixed two-stage normalization.

Preprocessing (added v5.2):
  - PE/PB clipping: rank computed on clip(pe, 0.1, 500), clip(pb, 0.1, 50)
  - Log transforms: np.log1p on vol, amount, total_mv (skewed distributions)
  - Winsorization: per-feature clip to [P1, P99] before Z-score normalization
  - T+5 label: label = sum of next 5 trading days' pct_chg (higher SNR)

Features (26 dims total):
  Temporal (19): 10 raw + 1 vwap_gap + 8 tech  (per-stock Z-score across T)
  Cross    (7) : 3 rank + rel_beta + 3 val rank (per-date Z-score across N, tiled)

Output: processed/v5_windows/{train,val}/{year}/{date}.pt
"""

import os, sys, gc
import numpy as np
import pandas as pd
import torch
from collections import defaultdict

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PARQUET_PATH = os.path.join(ROOT, "processed", "all_data.parquet")
INDEX_PATH = os.path.join(ROOT, "data", "market", "000300.SH.csv")
WINDOWS_DIR = os.path.join(ROOT, "processed", "v7_windows")

T = 60
LABEL_HORIZON = 1  # T+1 single-day return as label

BASIC_RAW = ["open", "high", "low", "close", "vol", "amount", "pct_chg",
             "turnover_rate", "volume_ratio", "total_mv"]

TECH = ["macd", "macd_signal", "rsi", "bb_width", "bb_pct", "mom_5", "mom_20", "vol_20"]

CROSS_COLS = ["pct_chg", "amount", "turnover_rate"]
VAL_COLS = ["pe", "pb", "circ_mv"]

# Features to log-transform (right-skewed, power-law distributed)
LOG_COLS = ["vol", "amount", "total_mv"]

# PE/PB clip ranges for rank computation
PE_CLIP = (0.1, 500.0)
PB_CLIP = (0.1, 50.0)

TRAIN_START = "20190102"; TRAIN_END = "20241231"
VAL_START = "20250102"; VAL_END = "20251231"

FLUSH_INTERVAL = 500
MEM_LIMIT_MB = 2000


def add_tech(df):
    c = df["close"].astype(float)
    e12 = c.ewm(span=12, adjust=False).mean()
    e26 = c.ewm(span=26, adjust=False).mean()
    df["macd"] = e12 - e26
    df["macd_signal"] = df["macd"].ewm(span=9, adjust=False).mean()
    d = c.diff()
    g = d.clip(lower=0)
    l = (-d).clip(lower=0)
    rs = g.ewm(alpha=1/14, adjust=False).mean() / (l.ewm(alpha=1/14, adjust=False).mean() + 1e-8)
    df["rsi"] = 100 - 100 / (1 + rs)
    m20 = c.rolling(20).mean()
    s20 = c.rolling(20).std()
    df["bb_width"] = 2 * s20 / (m20 + 1e-8)
    df["bb_pct"] = (c - (m20 - 2 * s20)) / (4 * s20 + 1e-8)
    df["mom_5"] = c / c.shift(5) - 1
    df["mom_20"] = c / c.shift(20) - 1
    df["vol_20"] = c.pct_change().rolling(20).std()
    return df


def normalize_temporal(arr):
    mean = arr.mean(axis=0, keepdims=True)
    std = arr.std(axis=0, keepdims=True) + 1e-8
    return (arr - mean) / std


def normalize_cross(arr):
    mean = arr.mean(axis=0, keepdims=True)
    std = arr.std(axis=0, keepdims=True) + 1e-8
    return (arr - mean) / std


def winsorize_2d(arr, p_low=1, p_high=99):
    """Clip per-column to [P_low, P_high] percentile range across all rows."""
    lo = np.percentile(arr, p_low, axis=0, keepdims=True)
    hi = np.percentile(arr, p_high, axis=0, keepdims=True)
    return np.clip(arr, lo, hi)


def rank_percentile(arr):
    order = np.argsort(np.argsort(arr))
    return order.astype(np.float32) / max(len(arr) - 1, 1)


def get_split(date_str):
    if TRAIN_START <= date_str <= TRAIN_END: return "train"
    if VAL_START <= date_str <= VAL_END: return "val"
    return None


def estimate_mb(buf):
    total = 0
    for v in buf.values():
        n = len(v["labels"])
        total += n * (T * 19 * 4 + 4 + 7 * 4)
    return total / (1024 * 1024)


def flush_dates(buf, keys):
    saved = 0
    for key in keys:
        if key not in buf: continue
        b = buf.pop(key)
        split, date = key
        n = len(b["labels"])
        if n == 0: continue
        out_dir = os.path.join(WINDOWS_DIR, split, date[:4])
        os.makedirs(out_dir, exist_ok=True)
        temporal = np.stack(b["temporal"], axis=0)
        torch.save({
            "temporal": torch.from_numpy(temporal).float(),
            "labels": torch.tensor(b["labels"], dtype=torch.float32),
            "ts_codes": b["ts_codes"],
            "cross_raw": torch.from_numpy(np.stack(b["cross_raw"], axis=0)).float(),
        }, os.path.join(out_dir, f"{date}.pt"))
        saved += 1
    gc.collect()
    return saved


def main():
    os.makedirs(WINDOWS_DIR, exist_ok=True)

    print("[v5 build] Loading parquet ...")
    df = pd.read_parquet(PARQUET_PATH)
    df["trade_date"] = df["trade_date"].astype(str)
    for col in BASIC_RAW + TECH + VAL_COLS + ["vwap"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    print(f"[v5 build] {len(df)} rows, {df['ts_code'].nunique()} stocks")

    print("[v5 build] Loading CSI300 index ...")
    idx = pd.read_csv(INDEX_PATH, dtype={"trade_date": str})
    idx_pct_map = dict(zip(idx["trade_date"], idx["pct_chg"].astype(float)))

    stocks = sorted(df["ts_code"].unique())
    total = len(stocks)

    # ---- Pass 1: Per-stock temporal windows + cross_raw ----
    date_buf = defaultdict(lambda: {"temporal": [], "labels": [], "ts_codes": [], "cross_raw": []})

    tech_cols = BASIC_RAW + TECH
    cross_raw_cols = CROSS_COLS + VAL_COLS
    log_indices = [tech_cols.index(c) for c in LOG_COLS if c in tech_cols]

    print(f"[v5 build] T={T}, label=T+{LABEL_HORIZON}, log_cols={LOG_COLS}")

    for s_idx, ts_code in enumerate(stocks):
        sdf = df[df["ts_code"] == ts_code].sort_values("trade_date").reset_index(drop=True)
        sdf = add_tech(sdf)

        if len(sdf) < T + LABEL_HORIZON + 1:
            continue

        vals_temporal = sdf[tech_cols].values.astype(np.float32)
        vwap_vals = sdf["vwap"].values.astype(np.float32)
        close_vals = sdf["close"].values.astype(np.float32)
        cross_vals = sdf[cross_raw_cols].values.astype(np.float32)
        pct_vals = sdf["pct_chg"].values.astype(np.float32)

        for i in range(T - 1, len(sdf) - LABEL_HORIZON):
            trade_date = sdf["trade_date"].iloc[i]
            split = get_split(trade_date)
            if split is None: continue

            window_basic = vals_temporal[i - T + 1 : i + 1].copy()
            if np.isnan(window_basic).any(): continue

            # Log transforms for skewed features
            for li in log_indices:
                window_basic[:, li] = np.log1p(np.maximum(window_basic[:, li], 0))

            w_close = close_vals[i - T + 1 : i + 1]
            w_vwap = vwap_vals[i - T + 1 : i + 1]
            vwap_gap = w_close / np.maximum(w_vwap, 1e-8) - 1
            if np.isnan(vwap_gap).any(): continue

            temporal = np.concatenate([window_basic, vwap_gap[:, None]], axis=1)

            # T+N label: cumulative return over next LABEL_HORIZON days
            future = pct_vals[i + 1 : i + 1 + LABEL_HORIZON]
            if np.isnan(future).any(): continue
            label = float(np.sum(future))
            if np.isnan(label): continue

            cr = np.zeros(7, dtype=np.float32)
            cr[0] = cross_vals[i, 0]   # pct_chg
            cr[1] = cross_vals[i, 1]   # amount
            cr[2] = cross_vals[i, 2]   # turnover_rate
            cr[3] = idx_pct_map.get(trade_date, 0.0)  # idx_ret
            cr[4] = cross_vals[i, 3]   # pe
            cr[5] = cross_vals[i, 4]   # pb
            cr[6] = cross_vals[i, 5]   # circ_mv

            date_buf[(split, trade_date)]["temporal"].append(temporal)
            date_buf[(split, trade_date)]["labels"].append(label)
            date_buf[(split, trade_date)]["ts_codes"].append(ts_code)
            date_buf[(split, trade_date)]["cross_raw"].append(cr)

        if (s_idx + 1) % FLUSH_INTERVAL == 0:
            mb = estimate_mb(date_buf)
            print(f"  stock {s_idx+1}/{total} | buf {mb:.0f}MB | dates {len(date_buf)}")
            if mb > MEM_LIMIT_MB:
                flush_dates(date_buf, sorted(date_buf.keys())[:len(date_buf)//3])

    flush_dates(date_buf, sorted(date_buf.keys()))
    print(f"[v5 build] Pass 1 complete. Dates flushed.")

    # ---- Pass 2: Winsorization → cross features → normalize → save ----
    print("[v5 build] Pass 2: winsorization + cross features + normalize ...")
    all_pt = []
    for root, dirs, files in os.walk(WINDOWS_DIR):
        for f in files:
            if f.endswith(".pt"):
                all_pt.append(os.path.join(root, f))
    all_pt = sorted(all_pt)

    for i, pt_path in enumerate(all_pt):
        data = torch.load(pt_path, map_location="cpu", weights_only=True)
        temporal_raw = data["temporal"].numpy()    # [N, T, 19]
        labels = data["labels"].numpy()
        ts_codes = data["ts_codes"]
        cross_raw = data["cross_raw"].numpy()       # [N, 7]

        N = len(labels)
        if N < 2: continue

        # ---- 0. Winsorize temporal features per-date across N ----
        temporal_flat = temporal_raw.reshape(-1, temporal_raw.shape[-1])  # [N*T, 19]
        temporal_flat = winsorize_2d(temporal_flat, p_low=1, p_high=99)
        temporal_raw = temporal_flat.reshape(N, T, -1)

        # ---- 1. Normalize temporal features per stock across T ----
        temporal_norm = np.zeros_like(temporal_raw)
        for j in range(N):
            temporal_norm[j] = normalize_temporal(temporal_raw[j])

        # ---- 2. Compute cross-sectional features (with PE/PB clipping) ----
        cross_feat = np.zeros((N, 7), dtype=np.float32)

        for col_idx in [0, 1, 2, 4, 5, 6]:
            raw_vals = cross_raw[:, col_idx].copy()
            valid = ~np.isnan(raw_vals)
            if valid.sum() < 2:
                cross_feat[:, col_idx] = 0.0
            else:
                # Clip PE and PB before ranking
                if col_idx == 4:   # PE
                    raw_vals = np.clip(raw_vals, PE_CLIP[0], PE_CLIP[1])
                elif col_idx == 5:  # PB
                    raw_vals = np.clip(raw_vals, PB_CLIP[0], PB_CLIP[1])
                ranks = np.zeros(N, dtype=np.float32)
                valid_clip = ~np.isnan(raw_vals)
                if valid_clip.sum() >= 2:
                    ranks[valid_clip] = rank_percentile(raw_vals[valid_clip])
                cross_feat[:, col_idx] = ranks

        cross_feat[:, 3] = cross_raw[:, 0] - cross_raw[:, 3]  # rel_beta

        # ---- 3. Normalize cross features across N stocks ----
        cross_norm = normalize_cross(cross_feat)

        # ---- 4. Tile cross to all T steps and concatenate ----
        cross_tiled = np.tile(cross_norm[:, np.newaxis, :], (1, T, 1))  # [N, T, 7]
        features_full = np.concatenate([temporal_norm, cross_tiled], axis=-1)

        torch.save({
            "features": torch.from_numpy(features_full).float(),
            "labels": torch.from_numpy(labels).float(),
            "ts_codes": ts_codes,
        }, pt_path)

        if (i + 1) % 300 == 0:
            print(f"  cross-section {i+1}/{len(all_pt)}")

    print(f"[v5 build] Done. {len(all_pt)} files in {WINDOWS_DIR}")


if __name__ == "__main__":
    main()
