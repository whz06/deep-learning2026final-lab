"""score_v7spatial.py — V7 GRU+SpatialAttention 全量打分 (独立脚本).

用法 (Windows PowerShell):
  & python.exe score_v7spatial.py --start 20260105 --end 20260529 --device cuda

中途中断直接重跑即可——已存在日期自动跳过。
"""
import os, sys, argparse, time, numpy as np, pandas as pd, torch

# ── Paths ──
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(SCRIPT_DIR)          # train_li/
sys.path.insert(0, ROOT)

CKPT = os.path.join(ROOT, "v7", "checkpoints",
                    "gru_spatial_v7_d32_K5_H128_L1_D0.2_lr0.0003_N1024.pt")
OUT  = os.path.join(ROOT, "v7", "results", "daily_scores_spatial_t1.parquet")
PARQUET = os.path.join(ROOT, "processed", "all_data.parquet")
CSI_P   = os.path.join(ROOT, "data", "market", "000300.SH.csv")

from v7.models.gru_spatial import GRURankerSpatial

# ── Constants ──
W, WARMUP, DIMS = 60, 300, 26
RAW  = ["open","high","low","close","vol","amount","pct_chg","turnover_rate","volume_ratio","total_mv"]
TECH = ["macd","macd_signal","rsi","bb_width","bb_pct","mom_5","mom_20","vol_20"]
LOG_COLS  = ["vol","amount","total_mv"]
PE_CLIP   = (0.1, 500.0)
PB_CLIP   = (0.1, 50.0)
os.makedirs(os.path.dirname(OUT), exist_ok=True)


# ═══════════ Feature Engineering (self-contained, no external deps) ═══════════

def add_tech(df):
    """Compute 8 technical indicators on a single-stock DataFrame (in-place)."""
    c = df["close"].astype(float)
    e12, e26 = c.ewm(span=12, adjust=False).mean(), c.ewm(span=26, adjust=False).mean()
    df["macd"] = e12 - e26
    df["macd_signal"] = df["macd"].ewm(span=9, adjust=False).mean()
    d = c.diff(); g = d.clip(lower=0); l = (-d).clip(lower=0)
    rs = g.ewm(alpha=1/14, adjust=False).mean() / (l.ewm(alpha=1/14, adjust=False).mean() + 1e-8)
    df["rsi"] = 100 - 100 / (1 + rs)
    m20, s20 = c.rolling(20).mean(), c.rolling(20).std()
    df["bb_width"] = 2 * s20 / (m20 + 1e-8)
    df["bb_pct"]   = (c - (m20 - 2*s20)) / (4 * s20 + 1e-8)
    df["mom_5"]  = c / c.shift(5)  - 1
    df["mom_20"] = c / c.shift(20) - 1
    df["vol_20"] = c.pct_change().rolling(20).std()
    return df


def winsorize(arr):
    lo = np.nanpercentile(arr, 1, axis=0, keepdims=True)
    hi = np.nanpercentile(arr, 99, axis=0, keepdims=True)
    return np.clip(arr, lo, hi)


def norm_t(arr):
    m = np.nanmean(arr, axis=0, keepdims=True)
    s = np.nanstd(arr, axis=0, keepdims=True) + 1e-8
    return (arr - m) / s


def norm_c(arr):
    m = np.nanmean(arr, axis=0, keepdims=True)
    s = np.nanstd(arr, axis=0, keepdims=True) + 1e-8
    return (arr - m) / s


def rank_pct(arr):
    """Percentile-rank an array, returning [0,1] floats. NaN-safe."""
    out = np.full(len(arr), 0.5, dtype=np.float32)
    valid = ~np.isnan(arr)
    if valid.sum() >= 2:
        order = np.argsort(np.argsort(arr[valid]))
        out[valid] = order.astype(np.float32) / (valid.sum() - 1)
    return out


# ═══════════ Main ═══════════

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", default="20260105")
    parser.add_argument("--end",   default="20260529")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=4096)
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    START, END = args.start, args.end
    print(f"[v7spatial] device={device}  {START} → {END}  batch={args.batch_size}")

    # ── 1. Load model ──
    model = GRURankerSpatial(input_dim=DIMS, hidden_size=128, num_layers=1,
                             dropout=0.2, d_proj=32, K=5).to(device).eval()
    state = torch.load(CKPT, map_location="cpu", weights_only=True)
    state = {k.replace("_orig_mod.", ""): v for k, v in state.items()}
    model.load_state_dict(state, strict=True)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[v7spatial] params={n_params:,}")

    # Warmup forward — fixes spatial-attention first-batch NaN on CUDA
    with torch.no_grad():
        _ = model(torch.randn(64, W, DIMS, device=device))
    if device.type == "cuda":
        torch.cuda.synchronize()
    print("[v7spatial] warmup OK")

    # ── 2. Load data ──
    t0 = time.time()
    df = pd.read_parquet(PARQUET)
    df["trade_date"] = df["trade_date"].astype(str)
    all_dates = sorted(df["trade_date"].unique())

    if START not in all_dates:
        print(f"ERROR: start={START} not in data. Available: {all_dates[0]}~{all_dates[-1]}")
        sys.exit(1)
    END = END if END in all_dates else all_dates[-1]

    warmup_cutoff = max(0, all_dates.index(START) - WARMUP)
    df = df[df["trade_date"] >= all_dates[warmup_cutoff]]
    print(f"[v7spatial] data: {df['trade_date'].min()}~{df['trade_date'].max()} "
          f"({df['ts_code'].nunique()} stocks, {time.time()-t0:.1f}s)")

    # CSI300
    csi = pd.read_csv(CSI_P, dtype={"trade_date": str})
    csi_map = dict(zip(csi["trade_date"], csi["pct_chg"].astype(np.float32)))

    # ── 3. Build temporal feature store (per-stock, once) ──
    print("[v7spatial] Building temporal feature store (groupby) ...")
    t0 = time.time()
    df = df.sort_values(["ts_code", "trade_date"]).reset_index(drop=True)

    temporal_list = []   # list of [T_rows, 19]  per stock
    lastval_list  = []   # list of [T_rows, 6]   per stock
    date_pos_list = []   # list of {date: row_idx}
    stock_ids     = []

    for ts, sdf in df.groupby("ts_code", sort=True):
        sdf = sdf.reset_index(drop=True)
        if len(sdf) < W:
            continue
        # ffill raw columns
        for col in RAW:
            if col in sdf.columns:
                sdf[col] = sdf[col].ffill()

        sdf = add_tech(sdf)

        # vwap_gap with fallback
        if "vwap" in sdf.columns:
            sdf["vwap_gap"] = sdf["close"].astype(float) / sdf["vwap"].astype(float) - 1
        else:
            sdf["vwap_gap"] = 0.0

        # Temporal features: RAW + TECH + vwap_gap → 19 dims
        raw_a  = sdf[RAW].astype(np.float32).values
        tech_a = sdf[TECH].astype(np.float32).values
        vwap_a = sdf["vwap_gap"].astype(np.float32).values.reshape(-1, 1)
        temp   = np.concatenate([raw_a, tech_a, vwap_a], axis=1)

        # log1p for volume/amount/market-cap columns
        for ci, col in enumerate(RAW):
            if col in LOG_COLS:
                temp[:, ci] = np.log1p(np.maximum(temp[:, ci], 0))

        temporal_list.append(temp)

        # Last-day values for cross-sectional features
        lv = np.column_stack([
            sdf["pct_chg"].astype(np.float32).values,
            sdf["amount"].astype(np.float32).values,
            sdf["turnover_rate"].astype(np.float32).values
            if "turnover_rate" in sdf.columns else np.zeros(len(sdf), dtype=np.float32),
            sdf["pe"].astype(np.float32).values
            if "pe" in sdf.columns else np.zeros(len(sdf), dtype=np.float32),
            sdf["pb"].astype(np.float32).values
            if "pb" in sdf.columns else np.zeros(len(sdf), dtype=np.float32),
            sdf["circ_mv"].astype(np.float32).values
            if "circ_mv" in sdf.columns else np.zeros(len(sdf), dtype=np.float32),
        ])
        lastval_list.append(lv)
        date_pos_list.append({d: i for i, d in enumerate(sdf["trade_date"].tolist())})
        stock_ids.append(ts)

    n_stocks = len(stock_ids)
    print(f"[v7spatial] {n_stocks} stocks indexed ({time.time()-t0:.1f}s)")

    # ── 4. Determine dates to score (skip existing) ──
    target_dates = [d for d in all_dates if START <= d <= END]
    existing = pd.read_parquet(OUT) if os.path.exists(OUT) else None
    if existing is not None and len(existing) > 0:
        scored = set(existing["trade_date"].unique())
        target_dates = [d for d in target_dates if d not in scored]
        print(f"[v7spatial] existing={len(scored)} dates, new={len(target_dates)}")
    else:
        print(f"[v7spatial] target={len(target_dates)} dates")

    if not target_dates:
        print("[v7spatial] All done — nothing to score.")
        return

    # ── 5. Score each date ──
    all_rows = []
    total_start = time.time()

    for di, date in enumerate(target_dates):
        t_date = time.time()

        # Find valid stocks for this date (have >= W rows before date)
        valid_idx, positions = [], []
        for i in range(n_stocks):
            pos = date_pos_list[i].get(date)
            if pos is not None and pos >= W - 1:
                valid_idx.append(i)
                positions.append(pos)

        N = len(valid_idx)
        if N < 10:
            continue

        valid_idx = np.array(valid_idx, dtype=np.int32)
        positions = np.array(positions, dtype=np.int32)

        # ── Build feature tensor ──
        # Temporal: extract W-length windows
        tw = np.zeros((N, W, 19), dtype=np.float32)
        lv_arr = np.zeros((N, 6), dtype=np.float32)
        for j, (si, pos) in enumerate(zip(valid_idx, positions)):
            tw[j] = temporal_list[si][pos - W + 1 : pos + 1]
            lv_arr[j] = lastval_list[si][pos]

        # Winsorize + normalize temporal features
        tw = norm_t(winsorize(tw))
        tw = np.nan_to_num(tw, nan=0.0)  # cuDNN GRU: NaN in any batch element → entire batch NaN

        # Cross-sectional features (4 dims)
        pct_a = lv_arr[:, 0]
        amt_a = lv_arr[:, 1]
        tvr_a = lv_arr[:, 2]
        cross = np.column_stack([
            rank_pct(pct_a),
            rank_pct(amt_a),
            rank_pct(tvr_a),
            pct_a - np.float32(csi_map.get(date, 0.0)),
        ])
        cross = norm_c(cross)
        cross_t = np.tile(cross[:, np.newaxis, :], (1, W, 1))

        # Valuation features (3 dims)
        pe_a = lv_arr[:, 3]
        pb_a = lv_arr[:, 4]
        cm_a = lv_arr[:, 5]
        val = np.column_stack([
            rank_pct(np.clip(pe_a, *PE_CLIP)),
            rank_pct(np.clip(pb_a, *PB_CLIP)),
            rank_pct(cm_a),
        ])
        val = norm_c(val)
        val_t = np.tile(val[:, np.newaxis, :], (1, W, 1))

        # Final: 19 temporal + 4 cross + 3 val = 26 dims
        feats = np.concatenate([tw, cross_t, val_t], axis=2)  # [N, W, 26]

        # ── Batched inference ──
        BS = args.batch_size
        scores_all = []
        for s in range(0, N, BS):
            e = min(s + BS, N)
            bx = torch.from_numpy(feats[s:e]).float().to(device)
            with torch.no_grad():
                sc = model(bx)
            scores_all.append(sc.cpu().numpy())
        scores = np.concatenate(scores_all)

        for j, si in enumerate(valid_idx):
            all_rows.append({
                "trade_date": date,
                "ts_code": stock_ids[si],
                "score": float(scores[j]),
            })

        elapsed = time.time() - t_date
        if (di + 1) % 5 == 0 or di == 0 or di == len(target_dates) - 1:
            print(f"  [{di+1:>3}/{len(target_dates)}] {date}: {N} stocks  "
                  f"μ={scores.mean():.4f} σ={scores.std():.4f}  {elapsed:.1f}s")

    # ── 6. Save ──
    new_df = pd.DataFrame(all_rows)
    if existing is not None and len(existing) > 0:
        existing = existing[~existing["trade_date"].isin(new_df["trade_date"].unique())]
        final = pd.concat([existing, new_df], ignore_index=True)
    else:
        final = new_df

    final.to_parquet(OUT, index=False)
    total_elapsed = time.time() - total_start
    print(f"[v7spatial] Done. {len(final):,} rows → {OUT}  ({total_elapsed/60:.1f} min)")


if __name__ == "__main__":
    main()
