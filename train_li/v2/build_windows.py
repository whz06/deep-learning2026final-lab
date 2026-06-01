"""
v2/build_windows.py — Build feature-rich windows: raw OHLCV + tech indicators + cross-sectional.

Output: processed/v2_windows/{train,val}/{year}/{date}.pt
  Each .pt: {"features": [N, T, F], "labels": [N], "ts_codes": [N]}   (Z-score normalized)
"""
import os, sys, gc
import numpy as np
import pandas as pd
import torch
from collections import defaultdict

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PARQUET_PATH  = os.path.join(ROOT, "processed", "all_data.parquet")
INDEX_PATH    = os.path.join(ROOT, "data", "market", "000300.SH.csv")
WINDOWS_DIR   = os.path.join(ROOT, "processed", "v2_windows")

WINDOW_SIZE  = 60
RAW_FEATURES = ["open", "high", "low", "close", "vol", "amount", "pct_chg",
                "turnover_rate", "volume_ratio", "total_mv"]
LABEL_COL    = "pct_chg"

TRAIN_START = "20190102"; TRAIN_END = "20241231"
VAL_START   = "20250102"; VAL_END   = "20251231"

# ---- Technical indicators (vectorised, per-stock full series) ----

def add_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Compute MACD, RSI, BB, momentum, volatility. Modifies df in-place."""
    close = df["close"].astype(float)

    # MACD
    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    df["macd"] = ema12 - ema26
    df["macd_signal"] = df["macd"].ewm(span=9, adjust=False).mean()

    # RSI(14)
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = (-delta).clip(lower=0)
    avg_gain = gain.ewm(alpha=1/14, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/14, adjust=False).mean()
    rs = avg_gain / (avg_loss + 1e-8)
    df["rsi"] = 100 - 100 / (1 + rs)

    # Bollinger Bands (20, 2)
    ma20 = close.rolling(20).mean()
    std20 = close.rolling(20).std()
    df["bb_width"] = (2 * std20) / (ma20 + 1e-8)
    df["bb_pct"] = (close - (ma20 - 2*std20)) / (4 * std20 + 1e-8)

    # Momentum
    df["mom_5"] = close / close.shift(5) - 1
    df["mom_20"] = close / close.shift(20) - 1

    # 20-day volatility
    df["vol_20"] = close.pct_change().rolling(20).std()

    return df


# ---- Cross-sectional features (per-date, all stocks) ----

def add_cross_features(day_df: pd.DataFrame, index_pct: float) -> pd.DataFrame:
    """Compute rank percentiles and relative beta for stocks on ONE trading day."""
    for col in ["pct_chg", "amount", "turnover_rate"]:
        if col in day_df.columns:
            day_df[f"{col}_rank"] = day_df[col].rank(pct=True)

    day_df["rel_beta"] = day_df["pct_chg"] - index_pct
    return day_df


# ---- Normalization ----

def normalize_window(arr: np.ndarray) -> np.ndarray:
    mean = arr.mean(axis=0, keepdims=True)
    std  = arr.std(axis=0, keepdims=True) + 1e-8
    return (arr - mean) / std


def get_split(date_str: str):
    if TRAIN_START <= date_str <= TRAIN_END: return "train"
    if VAL_START   <= date_str <= VAL_END:   return "val"
    return None


# ---- Flush helpers ----

FLUSH_INTERVAL = 500
MEM_LIMIT_MB = 2000

def estimate_mb(buf):
    total = 0
    for v in buf.values():
        total += len(v["labels"]) * (WINDOW_SIZE * 22 * 4 + 4)
    return total / (1024*1024)

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
        features = np.stack(b["features"], axis=0)
        torch.save({
            "features": torch.from_numpy(features).float(),
            "labels":   torch.tensor(b["labels"], dtype=torch.float32),
            "ts_codes": b["ts_codes"],
        }, os.path.join(out_dir, f"{date}.pt"))
        saved += 1
    gc.collect()
    return saved


# ---- Main ----

ALL_COLS = RAW_FEATURES + ["macd", "macd_signal", "rsi", "bb_width", "bb_pct",
                            "mom_5", "mom_20", "vol_20",
                            "pct_chg_rank", "amount_rank", "turnover_rate_rank", "rel_beta"]

def main():
    os.makedirs(WINDOWS_DIR, exist_ok=True)

    print("[v2 build] Loading parquet ...")
    df = pd.read_parquet(PARQUET_PATH)
    df["trade_date"] = df["trade_date"].astype(str)
    print(f"[v2 build] {len(df)} rows, {df['ts_code'].nunique()} stocks")

    # Load CSI300 for beta
    print("[v2 build] Loading CSI300 index ...")
    idx = pd.read_csv(INDEX_PATH, dtype={"trade_date": str})
    idx_pct_map = dict(zip(idx["trade_date"], idx["pct_chg"].astype(float)))
    print(f"[v2 build] CSI300 dates: {len(idx_pct_map)}")

    stocks = sorted(df["ts_code"].unique())
    total_stocks = len(stocks)

    # Step 1: Per-stock: compute tech indicators, build raw windows
    date_buf = defaultdict(lambda: {"features": [], "labels": [], "ts_codes": []})

    for s_idx, ts_code in enumerate(stocks):
        sdf = df[df["ts_code"] == ts_code].sort_values("trade_date").reset_index(drop=True)
        sdf = add_technical_indicators(sdf)
        vals = sdf[RAW_FEATURES + ["macd","macd_signal","rsi","bb_width","bb_pct",
                                    "mom_5","mom_20","vol_20"]].values.astype(np.float32)

        if len(sdf) < WINDOW_SIZE + 1:
            continue

        for i in range(WINDOW_SIZE - 1, len(sdf) - 1):
            window = vals[i - WINDOW_SIZE + 1 : i + 1]      # [T, F_raw+tech]
            label  = sdf[LABEL_COL].iloc[i + 1]
            if np.isnan(window).any() or np.isnan(label):
                continue
            trade_date = sdf["trade_date"].iloc[i]
            split = get_split(trade_date)
            if split is None: continue
            date_buf[(split, trade_date)]["features"].append(window)
            date_buf[(split, trade_date)]["labels"].append(float(label))
            date_buf[(split, trade_date)]["ts_codes"].append(ts_code)

        if (s_idx + 1) % FLUSH_INTERVAL == 0:
            mb = estimate_mb(date_buf)
            print(f"  stock {s_idx+1}/{total_stocks} | buf {mb:.0f}MB | dates {len(date_buf)}")
            if mb > MEM_LIMIT_MB:
                flush_dates(date_buf, sorted(date_buf.keys())[:len(date_buf)//3])

    # Final flush of raw windows
    flush_dates(date_buf, sorted(date_buf.keys()))
    print(f"[v2 build] Raw windows built. {len(date_buf)} dates flushed.")

    # Step 2: Reload each date, add cross-sectional features, re-normalize, re-save
    print("[v2 build] Adding cross-sectional features ...")
    all_pt_files = []
    for root, dirs, files in os.walk(WINDOWS_DIR):
        for f in files:
            if f.endswith(".pt"):
                all_pt_files.append(os.path.join(root, f))

    for i, pt_path in enumerate(sorted(all_pt_files)):
        data = torch.load(pt_path, map_location="cpu", weights_only=True)
        features = data["features"].numpy()  # [N, T, 18]
        labels   = data["labels"].numpy()
        ts_codes = data["ts_codes"]

        # Build a mini DataFrame for cross-sectional ranking
        # Need pct_chg, amount, turnover_rate for each stock on this date
        # These are features[:, -1, 6] (pct_chg), features[:, -1, 4] (vol), features[:, -1, 8] (turnover_rate)
        # Actually they're at their respective indices in RAW_FEATURES
        pct_chg_col   = RAW_FEATURES.index("pct_chg")       # 6
        amount_col    = RAW_FEATURES.index("amount")         # 5
        turnover_col  = RAW_FEATURES.index("turnover_rate")  # 7

        # Compute rank percentiles
        pct_vals   = features[:, -1, pct_chg_col]
        amt_vals   = features[:, -1, amount_col]
        tnvr_vals  = features[:, -1, turnover_col]

        n_stocks = len(labels)
        pct_rank   = np.zeros(n_stocks, dtype=np.float32)
        amt_rank   = np.zeros(n_stocks, dtype=np.float32)
        tnvr_rank  = np.zeros(n_stocks, dtype=np.float32)

        # Rank using argsort
        for arr, out in [(pct_vals, pct_rank), (amt_vals, amt_rank), (tnvr_vals, tnvr_rank)]:
            order = np.argsort(np.argsort(arr))
            out[:] = order.astype(np.float32) / max(n_stocks - 1, 1)

        # Relative beta: last day's pct_chg - index pct_chg
        date_str = os.path.basename(pt_path).replace(".pt", "")
        idx_pct = idx_pct_map.get(date_str, 0.0)
        rel_beta = np.full(n_stocks, pct_vals - idx_pct, dtype=np.float32)

        # Expand cross features to [N, T, 4]
        cross_feat = np.stack([
            np.tile(pct_rank[:, None], (1, WINDOW_SIZE)),
            np.tile(amt_rank[:, None], (1, WINDOW_SIZE)),
            np.tile(tnvr_rank[:, None], (1, WINDOW_SIZE)),
            np.tile(rel_beta[:, None], (1, WINDOW_SIZE)),
        ], axis=-1)  # [N, T, 4]

        # Concatenate: [N, T, 18] + [N, T, 4] = [N, T, 22]
        features_full = np.concatenate([features, cross_feat], axis=-1)

        # Per-window Z-score normalize
        for j in range(n_stocks):
            features_full[j] = normalize_window(features_full[j])

        # Re-save
        torch.save({
            "features": torch.from_numpy(features_full).float(),
            "labels":   torch.from_numpy(labels).float(),
            "ts_codes": ts_codes,
        }, pt_path)

        if (i + 1) % 300 == 0:
            print(f"  cross-section {i+1}/{len(all_pt_files)}")

    print(f"[v2 build] Done. {len(all_pt_files)} files in {WINDOWS_DIR}")


if __name__ == "__main__":
    main()
