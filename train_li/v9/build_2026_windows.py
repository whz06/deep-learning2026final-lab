"""v9/build_2026_windows.py — Quick build of 2026 test windows using v7 feature pipeline.

Reuses v7/build_windows.py feature logic but only for 2026 dates.
"""
import os, sys, gc, time
import numpy as np, pandas as pd, torch
from collections import defaultdict

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PARQUET = os.path.join(ROOT, "processed", "all_data.parquet")
INDEX_P = os.path.join(ROOT, "data", "market", "000300.SH.csv")
OUT_DIR = os.path.join(ROOT, "processed", "v7_windows", "test2026")
os.makedirs(OUT_DIR, exist_ok=True)

T = 60; LABEL_HORIZON = 1
BASIC_RAW = ["open", "high", "low", "close", "vol", "amount", "pct_chg",
             "turnover_rate", "volume_ratio", "total_mv"]
TECH = ["macd", "macd_signal", "rsi", "bb_width", "bb_pct", "mom_5", "mom_20", "vol_20"]
CROSS_COLS = ["pct_chg", "amount", "turnover_rate"]
VAL_COLS = ["pe", "pb", "circ_mv"]
LOG_COLS = ["vol", "amount", "total_mv"]
PE_CLIP = (0.1, 500.0); PB_CLIP = (0.1, 50.0)
WINSOR_P = (1, 99)

t0 = time.time()
print(f"[build2026] Loading parquet...")
df = pd.read_parquet(PARQUET)
df["trade_date"] = df["trade_date"].astype(str)

# Get all unique dates for T=60 lookback from Jan 2026
all_dates = sorted(df["trade_date"].unique())
test_start_idx = all_dates.index("20260105") if "20260105" in all_dates else None
if test_start_idx is None:
    raise RuntimeError("20260105 not found in data")

# We need T+LABEL_HORIZON days per test date
test_dates = [d for d in all_dates if "2026" in d[:4] and d <= "20260529"]
print(f"[build2026] Test dates: {len(test_dates)} ({test_dates[0]} ~ {test_dates[-1]})")
print(f"[build2026] Lookback from index {test_start_idx}, need {(test_start_idx)} days before")

# Pre-compute CSI300 for beta
csi = pd.read_csv(INDEX_P, dtype={"trade_date": str})
csi_map = dict(zip(csi["trade_date"], pd.to_numeric(csi["pct_chg"], errors="coerce") / 100.0))

# Load all stock data into per-stock DataFrames
print(f"[build2026] Building per-stock DataFrames...")
df = df.sort_values(["ts_code", "trade_date"])

# Pre-filter to needed date range (T+LABEL_HORIZON before first test date)
first_idx = all_dates.index(test_dates[0])
needed_start = all_dates[max(0, first_idx - T)]
df = df[df["trade_date"] >= needed_start]
print(f"[build2026] Filtered to {len(df)} rows ({needed_start} onwards)")

# Group by stock
stocks = {}
for code, sdf in df.groupby("ts_code"):
    if len(sdf) < T + 2: continue
    sdf = sdf.sort_values("trade_date")
    stocks[code] = sdf

print(f"[build2026] {len(stocks)} stocks with >= T+2 days")
print(f"[build2026] Building windows...")

# Helper functions (from v7 build_windows.py)
def add_tech(sdf):
    c = sdf["close"].astype(float)
    e12 = c.ewm(span=12, adjust=False).mean()
    e26 = c.ewm(span=26, adjust=False).mean()
    sdf["macd"] = e12 - e26
    sdf["macd_signal"] = sdf["macd"].ewm(span=9, adjust=False).mean()
    d = c.diff()
    g = d.clip(lower=0); l = (-d).clip(lower=0)
    rs = g.rolling(14).mean() / (l.rolling(14).mean() + 1e-8)
    sdf["rsi"] = 100 - 100 / (1 + rs)
    ma20 = c.rolling(20).mean(); std20 = c.rolling(20).std()
    sdf["bb_width"] = (2 * std20) / (ma20 + 1e-8)
    sdf["bb_pct"] = (c - (ma20 - 2 * std20)) / (4 * std20 + 1e-8)
    sdf["mom_5"] = c / c.shift(5) - 1
    sdf["mom_20"] = c / c.shift(20) - 1
    sdf["vol_20"] = c.pct_change().rolling(20).std()
    return sdf

def normalize_temporal(arr):
    m, s = arr.mean(axis=0, keepdims=True), arr.std(axis=0, keepdims=True) + 1e-8
    return (arr - m) / s

def normalize_cross(arr):
    m, s = arr.mean(axis=0, keepdims=True), arr.std(axis=0, keepdims=True) + 1e-8
    return (arr - m) / s

def rank_pct(arr):
    valid = ~np.isnan(arr)
    out = np.zeros(len(arr), dtype=np.float32)
    if valid.sum() >= 2:
        order = np.argsort(np.argsort(arr[valid]))
        out[valid] = order.astype(np.float32) / max(valid.sum() - 1, 1)
    return out

def winsorize_2d(arr, p_low=1, p_high=99):
    lo = np.percentile(arr, p_low, axis=0, keepdims=True)
    hi = np.percentile(arr, p_high, axis=0, keepdims=True)
    return np.clip(arr, lo, hi)

# Process each test date
n_built = 0
for td in test_dates:
    td_idx = all_dates.index(td)
    # Need T days before td + label day
    needed_idx = td_idx - T
    if needed_idx < 0: continue

    needed_dates = all_dates[needed_idx:td_idx + LABEL_HORIZON]
    if len(needed_dates) < T + LABEL_HORIZON: continue

    features_list = []; codes_list = []; labels_list = []; vwap_gaps = []
    last_pct = []; last_amt = []; last_tvr = []; last_pe = []; last_pb = []; last_cm = []

    for code, sdf in stocks.items():
        sdf_sub = sdf[sdf["trade_date"].isin(needed_dates)]
        if len(sdf_sub) < T + LABEL_HORIZON: continue

        sdf_sub = sdf_sub.sort_values("trade_date")
        window = sdf_sub.iloc[:T + LABEL_HORIZON]

        # Label: next day return
        label = float(window.iloc[-1]["pct_chg"])

        # Features
        w = window.iloc[:T].copy()
        w = add_tech(w)

        # Raw temporal
        raw_vals = w[BASIC_RAW].astype(float).values  # [T, 10]
        # Tech
        tech_vals = w[TECH].astype(float).values      # [T, 8]
        # vwap_gap
        vwap_g = (w["close"].values / np.maximum(w.get("vwap", w["close"]).values, 1e-8) - 1)  # [T]
        # Log transforms
        for lc in LOG_COLS:
            if lc in BASIC_RAW:
                idx = BASIC_RAW.index(lc)
                raw_vals[:, idx] = np.log1p(np.maximum(raw_vals[:, idx], 0))

        temporal_all = np.concatenate([raw_vals, tech_vals, vwap_g[:, None]], axis=1)  # [T, 19]

        # Cross-sectional snapshots (from last row of window)
        last = w.iloc[-1]
        last_pct.append(float(last.get("pct_chg", 0)))
        last_amt.append(float(last.get("amount", 0)))
        last_tvr.append(float(last.get("turnover_rate", 0)))
        pe = float(last.get("pe", 0)); pb = float(last.get("pb", 0)); cm = float(last.get("circ_mv", 0))
        last_pe.append(pe); last_pb.append(pb); last_cm.append(cm)
        codes_list.append(code)
        labels_list.append(label)
        features_list.append(temporal_all)
        vwap_gaps.append(vwap_g)

    if len(codes_list) < 50: continue  # need enough stocks for cross-sectional

    N = len(codes_list)
    temporal_stack = np.stack(features_list, axis=0)  # [N, T, 19]

    # Cross-sectional features
    idx_pct_val = csi_map.get(td, 0)
    pct_arr = np.array(last_pct, dtype=np.float32)
    amt_arr = np.array(last_amt, dtype=np.float32)
    tvr_arr = np.array(last_tvr, dtype=np.float32)
    pe_arr = np.array(last_pe, dtype=np.float32)
    pb_arr = np.array(last_pb, dtype=np.float32)
    cm_arr = np.array(last_cm, dtype=np.float32)

    cross_feat = np.zeros((N, 7), dtype=np.float32)
    cross_feat[:, 0] = rank_pct(pct_arr)
    cross_feat[:, 1] = rank_pct(amt_arr)
    cross_feat[:, 2] = rank_pct(tvr_arr)
    cross_feat[:, 3] = pct_arr - idx_pct_val  # rel_beta
    cross_feat[:, 4] = rank_pct(np.clip(pe_arr, PE_CLIP[0], PE_CLIP[1]))
    cross_feat[:, 5] = rank_pct(np.clip(pb_arr, PB_CLIP[0], PB_CLIP[1]))
    cross_feat[:, 6] = rank_pct(cm_arr)
    cross_norm = normalize_cross(cross_feat)

    # Winsorize + normalize temporal
    Nt, Tt, Ft = temporal_stack.shape
    t_flat = temporal_stack.reshape(-1, Ft)
    t_flat = winsorize_2d(t_flat, WINSOR_P[0], WINSOR_P[1])
    temporal_stack = t_flat.reshape(Nt, Tt, Ft)

    final_feat = np.zeros((N, T, 26), dtype=np.float32)
    for i in range(N):
        t_norm = normalize_temporal(temporal_stack[i])  # [T, 19]
        cross_tiled = np.tile(cross_norm[i], (T, 1))    # [T, 7]
        final_feat[i] = np.concatenate([t_norm, cross_tiled], axis=-1)

    labels_arr = np.array(labels_list, dtype=np.float32)

    # Save
    out_path = os.path.join(OUT_DIR, f"{td}.pt")
    torch.save({
        "features": torch.from_numpy(final_feat),
        "labels": torch.from_numpy(labels_arr),
        "codes": codes_list
    }, out_path)

    n_built += 1
    if n_built % 10 == 0:
        elapsed = time.time() - t0
        print(f"  [{elapsed:.0f}s] {n_built}/{len(test_dates)}: {td}  N={N}")

print(f"\n[build2026] Done: {n_built} windows → {OUT_DIR}  [{time.time()-t0:.0f}s]")
