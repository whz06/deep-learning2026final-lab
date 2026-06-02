"""v8/build_windows_v8.py — T+1 label + 5 new features + industry ID.

New features (5):
  Tier 1: amihud_20, price_pos_20, ret_skew_20
  Moneyflow: mf_flow_hhi, mf_sm_lg_div

Industry: from basic.csv → industry_id per stock

Features (31 dims total):
  Temporal (24): 10 raw + 8 tech + 3 Tier1 + 2 moneyflow + 1 vwap_gap
  Cross    (7) : 3 rank + rel_beta + 3 val rank (per-date Z-score, tiled)

Output: processed/v8_windows/{train,val}/{year}/{date}.pt
"""
import os, sys, gc, glob
import numpy as np
import pandas as pd
import torch
from collections import defaultdict

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PARQUET_PATH = os.path.join(ROOT, "processed", "all_data.parquet")
MONEYFLOW_DIR = os.path.join(ROOT, "data", "moneyflow")
INDEX_PATH = os.path.join(ROOT, "data", "market", "000300.SH.csv")
BASIC_PATH = os.path.join(ROOT, "data", "basic.csv")
WINDOWS_DIR = os.path.join(ROOT, "processed", "v8_windows")

T = 60
LABEL_HORIZON = 1
N_TEMPORAL = 24

BASIC_RAW = ["open", "high", "low", "close", "vol", "amount", "pct_chg",
             "turnover_rate", "volume_ratio", "total_mv"]
TECH = ["macd", "macd_signal", "rsi", "bb_width", "bb_pct", "mom_5", "mom_20", "vol_20"]
NEW_TIER1 = ["amihud_20", "price_pos_20", "ret_skew_20"]
NEW_MF = ["mf_flow_hhi", "mf_sm_lg_div"]

TECH_COLS = BASIC_RAW + TECH + NEW_TIER1 + NEW_MF  # 23 columns
CROSS_COLS = ["pct_chg", "amount", "turnover_rate"]
VAL_COLS = ["pe", "pb", "circ_mv"]

LOG_COLS = ["vol", "amount", "total_mv"]
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
    g = d.clip(lower=0); l = (-d).clip(lower=0)
    rs = g.ewm(alpha=1/14, adjust=False).mean() / (l.ewm(alpha=1/14, adjust=False).mean() + 1e-8)
    df["rsi"] = 100 - 100 / (1 + rs)
    m20 = c.rolling(20).mean(); s20 = c.rolling(20).std()
    df["bb_width"] = 2 * s20 / (m20 + 1e-8)
    df["bb_pct"] = (c - (m20 - 2 * s20)) / (4 * s20 + 1e-8)
    df["mom_5"] = c / c.shift(5) - 1
    df["mom_20"] = c / c.shift(20) - 1
    df["vol_20"] = c.pct_change().rolling(20).std()
    return df


def add_new_features(sdf):
    """Add Tier 1 features: amihud_20, price_pos_20, ret_skew_20."""
    ret = sdf["pct_chg"].astype(float) / 100.0
    amount = sdf["amount"].astype(float)
    sdf["amihud_20"] = (np.abs(ret) / np.maximum(amount, 1e-8)).rolling(20, min_periods=5).mean()

    close = sdf["close"].astype(float)
    L20 = close.rolling(20, min_periods=5).min()
    H20 = close.rolling(20, min_periods=5).max()
    sdf["price_pos_20"] = (close - L20) / np.maximum(H20 - L20, 1e-8)

    sdf["ret_skew_20"] = ret.rolling(20, min_periods=10).skew()
    sdf["ret_skew_20"] = sdf["ret_skew_20"].fillna(0.0)
    return sdf


def compute_moneyflow_features(df):
    """Compute mf_flow_hhi and mf_sm_lg_div from moneyflow columns in-place."""
    mf_cols = ["buy_sm_vol","sell_sm_vol","buy_md_vol","sell_md_vol",
               "buy_lg_vol","sell_lg_vol","buy_elg_vol","sell_elg_vol"]
    has_all = all(c in df.columns for c in mf_cols)
    if not has_all:
        print("[v8] Warning: moneyflow columns missing, filling mf features with 0")
        df["mf_flow_hhi"] = 0.0
        df["mf_sm_lg_div"] = 0.0
        return df

    for c in mf_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0.0)

    sm_t = df["buy_sm_vol"] + df["sell_sm_vol"]
    md_t = df["buy_md_vol"] + df["sell_md_vol"]
    lg_t = df["buy_lg_vol"] + df["sell_lg_vol"]
    elg_t = df["buy_elg_vol"] + df["sell_elg_vol"]
    total = sm_t + md_t + lg_t + elg_t

    total_a = np.maximum(total.values, 1e-8)
    shares = np.stack([sm_t.values, md_t.values, lg_t.values, elg_t.values], axis=1) / total_a[:, None]
    df["mf_flow_hhi"] = (shares ** 2).sum(axis=1).astype(np.float32)

    sm_net = df["buy_sm_vol"] - df["sell_sm_vol"]
    lg_net = df["buy_lg_vol"] + df["buy_elg_vol"] - df["sell_lg_vol"] - df["sell_elg_vol"]
    df["mf_sm_lg_div"] = (sm_net - lg_net).astype(np.float32)
    return df


def load_moneyflow(start_date, end_date):
    """Load all moneyflow CSVs in date range, compute features, return df with (ts_code,trade_date,mf_flow_hhi,mf_sm_lg_div)."""
    pattern = os.path.join(MONEYFLOW_DIR, "*.csv")
    files = sorted(glob.glob(pattern))
    if not files:
        print("[v8] No moneyflow files found")
        return None
    frames = []
    for f in files:
        d = os.path.basename(f).replace(".csv","")
        if d < start_date or d > end_date:
            continue
        try:
            chunk = pd.read_csv(f, dtype={"ts_code": str, "trade_date": str})
            needed = ["ts_code","trade_date",
                      "buy_sm_vol","sell_sm_vol","buy_md_vol","sell_md_vol",
                      "buy_lg_vol","sell_lg_vol","buy_elg_vol","sell_elg_vol"]
            available = [c for c in needed if c in chunk.columns]
            if len(available) < 10:
                continue
            chunk = chunk[available]
            for c in available:
                if c not in ["ts_code","trade_date"]:
                    chunk[c] = pd.to_numeric(chunk[c], errors="coerce")
            frames.append(chunk)
        except Exception:
            continue
    if not frames:
        return None
    mf = pd.concat(frames, ignore_index=True)
    mf = compute_moneyflow_features(mf)
    return mf[["ts_code","trade_date","mf_flow_hhi","mf_sm_lg_div"]]


def normalize_temporal(arr):
    mean = arr.mean(axis=0, keepdims=True)
    std = arr.std(axis=0, keepdims=True) + 1e-8
    return (arr - mean) / std


def normalize_cross(arr):
    mean = arr.mean(axis=0, keepdims=True)
    std = arr.std(axis=0, keepdims=True) + 1e-8
    return (arr - mean) / std


def winsorize_2d(arr, p_low=1, p_high=99):
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
        total += n * (T * N_TEMPORAL * 4 + 4 + 7 * 4 + 4)  # +4 for industry_id
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
        dst_path = os.path.join(out_dir, f"{date}.pt")
        tmp_path = dst_path + ".tmp"
        temporal = np.stack(b["temporal"], axis=0)
        torch.save({
            "temporal": torch.from_numpy(temporal).float(),
            "labels": torch.tensor(b["labels"], dtype=torch.float32),
            "ts_codes": b["ts_codes"],
            "cross_raw": torch.from_numpy(np.stack(b["cross_raw"], axis=0)).float(),
            "industry_id": torch.tensor(b["industry_id"], dtype=torch.long),
        }, tmp_path)
        os.replace(tmp_path, dst_path)
        saved += 1
    gc.collect()
    return saved


def main():
    os.makedirs(WINDOWS_DIR, exist_ok=True)

    # ---- 0. Load industry mapping ----
    print("[v8] Loading industry mapping ...")
    basic = pd.read_csv(BASIC_PATH, dtype={"ts_code": str})
    basic["industry"] = basic["industry"].fillna("Other")
    all_industries = sorted(basic["industry"].unique())
    ind2id = {ind: i for i, ind in enumerate(all_industries)}
    ts2ind = dict(zip(basic["ts_code"], basic["industry"].map(ind2id)))
    n_ind = len(all_industries)
    default_ind = ind2id.get("Other", 0)
    print(f"[v8] {n_ind} industries loaded")

    # ---- 1. Load main data ----
    print("[v8] Loading all_data.parquet ...")
    df = pd.read_parquet(PARQUET_PATH)
    df["trade_date"] = df["trade_date"].astype(str)
    for col in BASIC_RAW + TECH + VAL_COLS + ["vwap","pct_chg","amount","turnover_rate"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    print(f"[v8] {len(df)} rows, {df['ts_code'].nunique()} stocks")

    # Merge moneyflow features
    print("[v8] Loading moneyflow data ...")
    mf_feats = load_moneyflow("20160101", "20251231")
    if mf_feats is not None:
        df = df.merge(mf_feats, on=["ts_code","trade_date"], how="left")
        df["mf_flow_hhi"] = df["mf_flow_hhi"].fillna(0.0)
        df["mf_sm_lg_div"] = df["mf_sm_lg_div"].fillna(0.0)
        print(f"[v8] Moneyflow merged: {len(df)} rows")
    else:
        df["mf_flow_hhi"] = 0.0
        df["mf_sm_lg_div"] = 0.0

    # ---- 2. Load CSI300 ----
    idx = pd.read_csv(INDEX_PATH, dtype={"trade_date": str})
    idx_pct_map = dict(zip(idx["trade_date"], idx["pct_chg"].astype(float)))

    stocks = sorted(df["ts_code"].unique())
    total = len(stocks)

    # ---- Pass 1: Per-stock windows ----
    date_buf = defaultdict(lambda: {"temporal": [], "labels": [], "ts_codes": [],
                                     "cross_raw": [], "industry_id": []})

    cross_raw_cols = CROSS_COLS + VAL_COLS
    log_indices = [TECH_COLS.index(c) for c in LOG_COLS if c in TECH_COLS]
    all_tech_cols = list(TECH_COLS)  # 23 cols before vwap_gap

    print(f"[v8] T={T}, label=T+{LABEL_HORIZON}, temporal_dim={N_TEMPORAL}, industries={n_ind}")

    for s_idx, ts_code in enumerate(stocks):
        sdf = df[df["ts_code"] == ts_code].sort_values("trade_date").reset_index(drop=True)
        sdf = add_tech(sdf)
        sdf = add_new_features(sdf)

        if len(sdf) < T + LABEL_HORIZON + 1:
            continue

        vals_temporal = sdf[all_tech_cols].values.astype(np.float32)  # [L, 23]
        vwap_vals = sdf["vwap"].values.astype(np.float32)
        close_vals = sdf["close"].values.astype(np.float32)
        cross_vals = sdf[cross_raw_cols].values.astype(np.float32)
        pct_vals = sdf["pct_chg"].values.astype(np.float32)
        industry_id = ts2ind.get(ts_code, default_ind)

        for i in range(T - 1, len(sdf) - LABEL_HORIZON):
            trade_date = sdf["trade_date"].iloc[i]
            split = get_split(trade_date)
            if split is None: continue

            window_basic = vals_temporal[i - T + 1 : i + 1].copy()  # [T, 23]
            if np.isnan(window_basic).any(): continue

            for li in log_indices:
                window_basic[:, li] = np.log1p(np.maximum(window_basic[:, li], 0))

            w_close = close_vals[i - T + 1 : i + 1]
            w_vwap = vwap_vals[i - T + 1 : i + 1]
            vwap_gap = w_close / np.maximum(w_vwap, 1e-8) - 1
            if np.isnan(vwap_gap).any(): continue

            temporal = np.concatenate([window_basic, vwap_gap[:, None]], axis=1)  # [T, 24]

            future = pct_vals[i + 1 : i + 1 + LABEL_HORIZON]
            if np.isnan(future).any(): continue
            label = float(np.sum(future))
            if np.isnan(label): continue

            cr = np.zeros(7, dtype=np.float32)
            cr[0] = cross_vals[i, 0]
            cr[1] = cross_vals[i, 1]
            cr[2] = cross_vals[i, 2]
            cr[3] = idx_pct_map.get(trade_date, 0.0)
            cr[4] = cross_vals[i, 3]
            cr[5] = cross_vals[i, 4]
            cr[6] = cross_vals[i, 5]

            date_buf[(split, trade_date)]["temporal"].append(temporal)
            date_buf[(split, trade_date)]["labels"].append(label)
            date_buf[(split, trade_date)]["ts_codes"].append(ts_code)
            date_buf[(split, trade_date)]["cross_raw"].append(cr)
            date_buf[(split, trade_date)]["industry_id"].append(industry_id)

        if (s_idx + 1) % FLUSH_INTERVAL == 0:
            mb = estimate_mb(date_buf)
            print(f"  stock {s_idx+1}/{total} | buf {mb:.0f}MB | dates {len(date_buf)}")
            if mb > MEM_LIMIT_MB:
                flush_dates(date_buf, sorted(date_buf.keys())[:len(date_buf)//3])

    flush_dates(date_buf, sorted(date_buf.keys()))
    print("[v8] Pass 1 complete.")

    # ---- Pass 2: Winsorization + cross features + normalize ----
    print("[v8] Pass 2: winsorization + cross features + normalize ...")
    all_pt = []
    for root, dirs, files in os.walk(WINDOWS_DIR):
        for f in files:
            if f.endswith(".pt"):
                all_pt.append(os.path.join(root, f))
    all_pt = sorted(all_pt)

    for i, pt_path in enumerate(all_pt):
        data = torch.load(pt_path, map_location="cpu", weights_only=True)
        temporal_raw = data["temporal"].numpy()    # [N, T, 24]
        labels = data["labels"].numpy()
        ts_codes = data["ts_codes"]
        cross_raw = data["cross_raw"].numpy()       # [N, 7]
        industry_id = data.get("industry_id", torch.zeros(len(labels), dtype=torch.long))

        N = len(labels)
        if N < 2: continue

        # Winsorize temporal features
        temporal_flat = temporal_raw.reshape(-1, N_TEMPORAL)
        temporal_flat = winsorize_2d(temporal_flat, p_low=1, p_high=99)
        temporal_raw = temporal_flat.reshape(N, T, -1)

        # Normalize temporal per stock across T
        temporal_norm = np.zeros_like(temporal_raw)
        for j in range(N):
            temporal_norm[j] = normalize_temporal(temporal_raw[j])

        # Cross-sectional features
        cross_feat = np.zeros((N, 7), dtype=np.float32)
        for col_idx in [0, 1, 2, 4, 5, 6]:
            raw_vals = cross_raw[:, col_idx].copy()
            valid = ~np.isnan(raw_vals)
            if valid.sum() < 2:
                cross_feat[:, col_idx] = 0.0
            else:
                if col_idx == 4:
                    raw_vals = np.clip(raw_vals, PE_CLIP[0], PE_CLIP[1])
                elif col_idx == 5:
                    raw_vals = np.clip(raw_vals, PB_CLIP[0], PB_CLIP[1])
                ranks = np.zeros(N, dtype=np.float32)
                valid_clip = ~np.isnan(raw_vals)
                if valid_clip.sum() >= 2:
                    ranks[valid_clip] = rank_percentile(raw_vals[valid_clip])
                cross_feat[:, col_idx] = ranks
        cross_feat[:, 3] = cross_raw[:, 0] - cross_raw[:, 3]

        cross_norm = normalize_cross(cross_feat)
        cross_tiled = np.tile(cross_norm[:, np.newaxis, :], (1, T, 1))
        features_full = np.concatenate([temporal_norm, cross_tiled], axis=-1)  # [N, T, 31]

        torch.save({
            "features": torch.from_numpy(features_full).float(),
            "labels": torch.from_numpy(labels).float(),
            "ts_codes": ts_codes,
            "industry_id": industry_id,
        }, pt_path + ".tmp")
        os.replace(pt_path + ".tmp", pt_path)

        if (i + 1) % 300 == 0:
            print(f"  cross-section {i+1}/{len(all_pt)}")

    print(f"[v8] Done. {len(all_pt)} files in {WINDOWS_DIR}")


if __name__ == "__main__":
    main()
