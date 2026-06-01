"""
v2/infer.py — Competition inference: predict stock scores for latest trading day.

Usage:
  python infer.py --model gru --ckpt gru_best.pt --top-n 20 --bottom-k 5
Outputs: buy_list.txt  sell_list.txt
"""
import os, sys, re
import numpy as np
import pandas as pd
import torch

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(SCRIPT_DIR)
PARQUET_PATH = os.path.join(ROOT, "processed", "all_data.parquet")
INDEX_PATH   = os.path.join(ROOT, "data", "market", "000300.SH.csv")

from config import WINDOW_SIZE
from models import MLPRanker, GRURanker, TransformerRanker

# Feature columns (must match build_windows.py)
RAW_FEATURES = ["open","high","low","close","vol","amount","pct_chg",
                "turnover_rate","volume_ratio","total_mv"]
TECH_FEATURES = ["macd","macd_signal","rsi","bb_width","bb_pct",
                 "mom_5","mom_20","vol_20"]
CROSS_FEATURES = ["pct_chg_rank","amount_rank","turnover_rate_rank","rel_beta"]


def add_technical_indicators(df):
    close = df["close"].astype(float)
    ema12, ema26 = close.ewm(span=12, adjust=False).mean(), close.ewm(span=26, adjust=False).mean()
    df["macd"] = ema12 - ema26
    df["macd_signal"] = df["macd"].ewm(span=9, adjust=False).mean()
    delta = close.diff()
    gain, loss = delta.clip(lower=0), (-delta).clip(lower=0)
    rs = gain.ewm(alpha=1/14, adjust=False).mean() / (loss.ewm(alpha=1/14, adjust=False).mean() + 1e-8)
    df["rsi"] = 100 - 100 / (1 + rs)
    ma20, std20 = close.rolling(20).mean(), close.rolling(20).std()
    df["bb_width"] = (2*std20) / (ma20 + 1e-8)
    df["bb_pct"] = (close - (ma20 - 2*std20)) / (4*std20 + 1e-8)
    df["mom_5"] = close / close.shift(5) - 1
    df["mom_20"] = close / close.shift(20) - 1
    df["vol_20"] = close.pct_change().rolling(20).std()
    return df


def normalize(arr):
    m, s = arr.mean(axis=0, keepdims=True), arr.std(axis=0, keepdims=True) + 1e-8
    return (arr - m) / s


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, choices=["mlp","gru","tf"])
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--top-n", type=int, default=20)
    parser.add_argument("--bottom-k", type=int, default=5)
    parser.add_argument("--date", type=str, default="",
                        help="Feature end date (default: latest trading date in data)")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Parse model params from checkpoint filename
    ckpt_name = args.ckpt.replace(".pt", "")
    if args.model == "gru":
        hs = int(re.search(r"hidden_size=(\d+)", ckpt_name).group(1))
        nl = int(re.search(r"num_layers=(\d+)", ckpt_name).group(1))
        do = float(re.search(r"dropout=([\d.]+)_lr", ckpt_name).group(1))
        model = GRURanker(input_dim=22, hidden_size=hs, num_layers=nl, dropout=do)
    elif args.model == "mlp":
        hd = int(re.search(r"hidden_dim=(\d+)", ckpt_name).group(1))
        nl = int(re.search(r"n_layers=(\d+)", ckpt_name).group(1))
        do = float(re.search(r"dropout=([\d.]+)_lr", ckpt_name).group(1))
        model = MLPRanker(input_dim=22*60, hidden_dim=hd, n_layers=nl, dropout=do)
    else:  # tf
        dm = int(re.search(r"d_model=(\d+)", ckpt_name).group(1))
        nh = int(re.search(r"n_heads=(\d+)", ckpt_name).group(1))
        nt = int(re.search(r"n_temporal_layers=(\d+)", ckpt_name).group(1))
        ns = int(re.search(r"n_spatial_layers=(\d+)", ckpt_name).group(1))
        do = float(re.search(r"dropout=([\d.]+)_lr", ckpt_name).group(1))
        model = TransformerRanker(input_dim=22, d_model=dm, n_heads=nh,
                                   n_temporal_layers=nt, n_spatial_layers=ns, dropout=do)
    ckpt_path = os.path.join(SCRIPT_DIR, "checkpoints", args.ckpt)
    model.load_state_dict(torch.load(ckpt_path, map_location=device, weights_only=True))
    model.to(device)
    model.eval()

    print("[infer] Loading data ...")
    df = pd.read_parquet(PARQUET_PATH)
    df["trade_date"] = df["trade_date"].astype(str)
    latest_date = args.date if args.date else sorted(df["trade_date"].unique())[-1]
    print(f"[infer] Feature end date: {latest_date}")

    # Keep ~150 days for technical indicator warmup (MACD needs 26, BB needs 20)
    all_dates_sorted = sorted(df["trade_date"].unique())
    date_idx = all_dates_sorted.index(latest_date)
    cutoff_idx = max(0, date_idx - 150)
    df = df[df["trade_date"].isin(all_dates_sorted[cutoff_idx:])]
    print(f"[infer] Trimmed to {len(df)} rows ({date_idx-cutoff_idx} days)")

    # Load CSI300
    idx = pd.read_csv(INDEX_PATH, dtype={"trade_date": str})
    idx_map = dict(zip(idx["trade_date"], idx["pct_chg"].astype(float)))
    idx_pct = idx_map.get(latest_date, 0.0)

    # Group by stock for O(N log N) processing instead of per-stock O(N²) filtering
    print(f"[infer] Computing features (groupby mode) ...")
    grouped = df.groupby("ts_code")
    windows = {}
    last_pct, last_amt, last_tvr = {}, {}, {}

    for ts_code, sdf in grouped:
        sdf = sdf.sort_values("trade_date")
        sdf = sdf[sdf["trade_date"] <= latest_date]
        if len(sdf) < WINDOW_SIZE:
            continue
        # Compute tech indicators on full history (more accurate rolling/ewm)
        sdf = add_technical_indicators(sdf)
        # Take last 60 rows as feature window
        window_data = sdf[RAW_FEATURES + TECH_FEATURES].values.astype(np.float32)[-WINDOW_SIZE:]
        if np.isnan(window_data).any():
            continue
        windows[ts_code] = window_data
        last_row = sdf.iloc[-1]           # latest_date row
        last_pct[ts_code] = last_row["pct_chg"]
        last_amt[ts_code] = last_row["amount"]
        last_tvr[ts_code] = last_row["turnover_rate"]

    # Cross-sectional ranks
    code_list = list(windows.keys())
    n = len(code_list)
    pct_arr = np.array([last_pct[c] for c in code_list])
    amt_arr = np.array([last_amt[c] for c in code_list])
    tvr_arr = np.array([last_tvr[c] for c in code_list])

    pct_rank = np.argsort(np.argsort(pct_arr)).astype(np.float32) / max(n-1, 1)
    amt_rank = np.argsort(np.argsort(amt_arr)).astype(np.float32) / max(n-1, 1)
    tvr_rank = np.argsort(np.argsort(tvr_arr)).astype(np.float32) / max(n-1, 1)
    rel_beta_vals = np.full(n, pct_arr - idx_pct, dtype=np.float32)

    # Build final features and predict
    batch_feat = []
    batch_code = []
    for i, ts_code in enumerate(code_list):
        win = windows[ts_code]                                 # [T, 18]
        cross = np.column_stack([
            np.full(WINDOW_SIZE, pct_rank[i]),
            np.full(WINDOW_SIZE, amt_rank[i]),
            np.full(WINDOW_SIZE, tvr_rank[i]),
            np.full(WINDOW_SIZE, rel_beta_vals[i]),
        ]).astype(np.float32)                                   # [T, 4]
        feat = np.concatenate([win, cross], axis=-1)            # [T, 22]
        feat = normalize(feat)
        batch_feat.append(feat)
        batch_code.append(ts_code)

    print(f"[infer] Predicting {len(batch_feat)} stocks ...")
    batch_tensor = torch.from_numpy(np.stack(batch_feat)).to(device)

    with torch.no_grad():
        all_scores = model(batch_tensor).cpu().numpy()

    # Sort and output
    idx_sorted = np.argsort(all_scores)[::-1]
    top_n = idx_sorted[:args.top_n]
    bottom_k = idx_sorted[-args.bottom_k:]

    print(f"\n=== BUY (top {args.top_n}) ===")
    for i in top_n:
        print(f"  {batch_code[i]:12s}  score={all_scores[i]:.4f}")
    print(f"\n=== SELL (bottom {args.bottom_k}) ===")
    for i in bottom_k:
        print(f"  {batch_code[i]:12s}  score={all_scores[i]:.4f}")

    with open(os.path.join(SCRIPT_DIR, "buy_list.txt"), "w") as f:
        f.write("\n".join(batch_code[i] for i in top_n))
    with open(os.path.join(SCRIPT_DIR, "sell_list.txt"), "w") as f:
        f.write("\n".join(batch_code[i] for i in bottom_k))
    print("[infer] Saved buy_list.txt / sell_list.txt")


if __name__ == "__main__":
    main()
