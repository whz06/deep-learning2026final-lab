"""
trade/infer.py — Daily competition inference with Strategy B (momentum stop-loss).

Usage:
  python infer.py --date 20260529 --top-n 20 --bottom-k 5
  python infer.py --date 20260529 --dry-run    # decision only, no file writes

Outputs (in trade/):
  buy_list.txt   — top-N stock codes for next-day buy
  sell_list.txt  — bottom-K stock codes for next-day sell
  decision.log   — daily position decision log (append-only)

Strategy B: when CSI300 5-day cumulative return < -1%, reduce position to 80%.
"""
import os, sys, re, argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(SCRIPT_DIR)
PARQUET_PATH = os.path.join(ROOT, "processed", "all_data.parquet")
INDEX_PATH   = os.path.join(ROOT, "data", "market", "000300.SH.csv")
CKPT_DIR     = os.path.join(ROOT, "v2", "checkpoints")
CKPT_DIR_V5  = os.path.join(ROOT, "v5", "checkpoints")

BUY_PATH  = os.path.join(SCRIPT_DIR, "buy_list.txt")
SELL_PATH = os.path.join(SCRIPT_DIR, "sell_list.txt")
LOG_PATH  = os.path.join(SCRIPT_DIR, "decision.log")

# ===========================================================
# Constants
# ===========================================================
WINDOW_SIZE = 60
WARMUP_DAYS = 150
RAW_FEATURES = ["open","high","low","close","vol","amount","pct_chg",
                "turnover_rate","volume_ratio","total_mv"]
TECH_FEATURES = ["macd","macd_signal","rsi","bb_width","bb_pct",
                 "mom_5","mom_20","vol_20"]
CROSS_COLS = ["pct_chg","amount","turnover_rate"]
VAL_COLS   = ["pe","pb","circ_mv"]
INPUT_DIM = 26  # 19 temporal (10 raw + 1 vwap_gap + 8 tech) + 7 cross (4 + 3 valuation)

# Strategy B defaults
STRATEGY_B_THRESHOLD = -1.0   # csi5d below this → risk-off
STRATEGY_B_MIN_POS   = 0.80   # minimum position (competition rule: >= 80%)

# ===========================================================
# GRU Model (inlined — identical to v2/models/gru.py)
# ===========================================================
class GRURanker(nn.Module):
    def __init__(self, input_dim, hidden_size=128, num_layers=2,
                 dropout=0.1, bidirectional=False):
        super().__init__()
        self.gru = nn.GRU(input_size=input_dim, hidden_size=hidden_size,
                          num_layers=num_layers, batch_first=True,
                          dropout=dropout if num_layers > 1 else 0.0,
                          bidirectional=bidirectional)
        dir_mult = 2 if bidirectional else 1
        self.head = nn.Sequential(
            nn.Linear(hidden_size * dir_mult, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, 1),
        )

    def forward(self, x):
        out, _ = self.gru(x)
        last = out[:, -1, :]
        return self.head(last).squeeze(-1)


# ===========================================================
# Feature computation
# ===========================================================
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


def normalize_temporal(arr):
    """Z-score per stock across T axis. arr: [T, F]"""
    m, s = arr.mean(axis=0, keepdims=True), arr.std(axis=0, keepdims=True) + 1e-8
    return (arr - m) / s


def normalize_cross(arr):
    """Z-score across N stocks. arr: [N, F]"""
    m, s = arr.mean(axis=0, keepdims=True), arr.std(axis=0, keepdims=True) + 1e-8
    return (arr - m) / s


# ===========================================================
# Strategy B: momentum stop-loss
# ===========================================================
def compute_csi5d(csi_series, target_date):
    """
    Compute CSI300 cumulative return over 5 trading days ending ON target_date.
    csi_series: pd.Series with trade_date index, pct_chg values.
    Returns: float (percentage)
    """
    csi_series = csi_series.sort_index()
    idx = csi_series.index.tolist()
    if target_date not in idx:
        # If target_date not in CSI300 data, find nearest prior date
        prior = [d for d in idx if d <= target_date]
        if not prior:
            return 0.0, 0.0
        target_date = prior[-1]
    pos = idx.index(target_date)
    start = max(0, pos - 4)  # 5 days: pos-4, pos-3, pos-2, pos-1, pos
    vals = csi_series.iloc[start:pos+1].values
    csi5d = float(np.sum(vals))
    return csi5d, float(csi_series.loc[target_date])


def decide_position(csi5d, threshold=STRATEGY_B_THRESHOLD, min_pos=STRATEGY_B_MIN_POS):
    """
    Returns (position_ratio, triggered: bool, reason: str)
    """
    if csi5d < threshold:
        return min_pos, True, f"CSI5d={csi5d:+.2f}% < {threshold:+.1f}% -> risk-off {min_pos*100:.0f}%"
    else:
        return 1.0, False, f"CSI5d={csi5d:+.2f}% >= {threshold:+.1f}% -> full position"


# ===========================================================
# Model loading
# ===========================================================
def load_model(ckpt_name, device, ckpt_dir, input_dim=26):
    ckpt_base = ckpt_name.replace(".pt", "")
    hs = int(re.search(r"hidden_size=(\d+)", ckpt_base).group(1)) if "hidden_size" in ckpt_base else \
         int(re.search(r"_H(\d+)_", ckpt_base).group(1))
    nl = int(re.search(r"num_layers=(\d+)", ckpt_base).group(1)) if "num_layers" in ckpt_base else \
         int(re.search(r"_L(\d+)_", ckpt_base).group(1))
    do = float(re.search(r"dropout=([\d.]+)_lr", ckpt_base).group(1)) if "_lr" in ckpt_base else \
         float(re.search(r"_D([\d.]+)$", ckpt_base).group(1))
    model = GRURanker(input_dim=input_dim, hidden_size=hs, num_layers=nl, dropout=do)
    ckpt_path = os.path.join(ckpt_dir, ckpt_name)
    model.load_state_dict(torch.load(ckpt_path, map_location=device, weights_only=True))
    model.to(device)
    model.eval()
    return model, hs, nl, do


# ===========================================================
# Main
# ===========================================================
def main():
    parser = argparse.ArgumentParser(description="Daily competition inference with Strategy B")
    parser.add_argument("--date", type=str, default="",
                        help="Feature cutoff date (YYYYMMDD). Default: latest in data.")
    parser.add_argument("--ckpt", type=str,
                        default="gru_v5_H128_L1_D0.2_lr0.0003_N2048.pt",
                        help="Checkpoint filename in v5/checkpoints/")
    parser.add_argument("--ckpt-dir", type=str, default="v5",
                        help="Checkpoint directory: v2 or v5")
    parser.add_argument("--top-n", type=int, default=20, help="Buy candidates at 100%% position")
    parser.add_argument("--bottom-k", type=int, default=5, help="Sell candidates at 100%% position")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dry-run", action="store_true", help="Print decision only, don't write files")
    parser.add_argument("--no-strategy", action="store_true", help="Disable Strategy B, always full position")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() and args.device == "cuda" else "cpu")
    print(f"[infer] Device: {device}")

    # ------------------------------------------------------------------
    # 1. Load model
    # ------------------------------------------------------------------
    ckpt_dir = CKPT_DIR_V5 if args.ckpt_dir == "v5" else CKPT_DIR
    input_dim = 26 if args.ckpt_dir == "v5" else 22
    model, hs, nl, do = load_model(args.ckpt, device, ckpt_dir, input_dim)
    print(f"[infer] Model: GRU H={hs} L={nl} D={do} input_dim={input_dim}")

    # ------------------------------------------------------------------
    # 2. Load data
    # ------------------------------------------------------------------
    print("[infer] Loading data ...")
    df = pd.read_parquet(PARQUET_PATH)
    df["trade_date"] = df["trade_date"].astype(str)
    dates_all = sorted(df["trade_date"].unique())

    latest_date = args.date if args.date else dates_all[-1]
    if latest_date not in dates_all:
        print(f"[infer] ERROR: date {latest_date} not in data. Available: {dates_all[0]} ~ {dates_all[-1]}")
        sys.exit(1)

    date_idx = dates_all.index(latest_date)
    cutoff_idx = max(0, date_idx - WARMUP_DAYS)
    df = df[df["trade_date"].isin(dates_all[cutoff_idx:])]
    print(f"[infer] Feature end date: {latest_date}  (history: {date_idx - cutoff_idx} days)")

    # ------------------------------------------------------------------
    # 3. CSI300 for Strategy B
    # ------------------------------------------------------------------
    csi = pd.read_csv(INDEX_PATH, dtype={"trade_date": str})
    csi["trade_date"] = csi["trade_date"].astype(str)
    idx_map = dict(zip(csi["trade_date"], csi["pct_chg"].astype(float)))
    idx_pct = idx_map.get(latest_date, 0.0)

    csi_series = pd.Series(idx_map.values(), index=list(idx_map.keys()))
    csi5d, csi_today = compute_csi5d(csi_series, latest_date)

    # ------------------------------------------------------------------
    # 4. Strategy B: position decision
    # ------------------------------------------------------------------
    if args.no_strategy:
        position = 1.0
        triggered = False
        reason = "Strategy B disabled (--no-strategy)"
    else:
        position, triggered, reason = decide_position(csi5d)

    buy_n = max(1, int(np.ceil(args.top_n * position)))
    sell_k = max(1, int(np.floor(args.bottom_k * position)))

    print(f"\n{'='*55}")
    print(f"  STRATEGY B: {reason}")
    print(f"  Position: {position*100:.0f}%  →  buy_n={buy_n}/{args.top_n}  sell_k={sell_k}/{args.bottom_k}")
    print(f"{'='*55}")

    # ------------------------------------------------------------------
    # 5. Feature computation (v5: fixed 2-stage normalization, 26 dims)
    # ------------------------------------------------------------------
    print(f"[infer] Computing features ...")
    grouped = df.groupby("ts_code")
    windows = {}
    last_pct, last_amt, last_tvr = {}, {}, {}
    last_pe, last_pb, last_cm = {}, {}, {}
    temporal_dim = len(RAW_FEATURES) + len(TECH_FEATURES) + 1  # +1 for vwap_gap

    for ts_code, sdf in grouped:
        sdf = sdf.sort_values("trade_date")
        sdf = sdf[sdf["trade_date"] <= latest_date]
        if len(sdf) < WINDOW_SIZE:
            continue
        sdf = add_technical_indicators(sdf)
        vals_rawtech = sdf[RAW_FEATURES + TECH_FEATURES].values.astype(np.float32)[-WINDOW_SIZE:]

        # vwap_gap: close/vwap - 1
        w_close = sdf["close"].values.astype(np.float32)[-WINDOW_SIZE:]
        w_vwap = sdf["vwap"].values.astype(np.float32)[-WINDOW_SIZE:]
        vwap_gap = w_close / np.maximum(w_vwap, 1e-8) - 1

        if np.isnan(vals_rawtech).any() or np.isnan(vwap_gap).any():
            continue

        windows[ts_code] = (vals_rawtech, vwap_gap)
        last_row = sdf.iloc[-1]
        last_pct[ts_code] = last_row["pct_chg"]
        last_amt[ts_code] = last_row["amount"]
        last_tvr[ts_code] = last_row["turnover_rate"]
        last_pe[ts_code] = last_row.get("pe", 0)
        last_pb[ts_code] = last_row.get("pb", 0)
        last_cm[ts_code] = last_row.get("circ_mv", 0)

    code_list = list(windows.keys())
    n_stocks = len(code_list)
    if n_stocks == 0:
        print("[infer] ERROR: No valid stocks found.")
        sys.exit(1)

    print(f"[infer] Valid stocks: {n_stocks}")

    # ---- Cross-sectional raw values ----
    pct_arr = np.array([last_pct[c] for c in code_list], dtype=np.float32)
    amt_arr = np.array([last_amt[c] for c in code_list], dtype=np.float32)
    tvr_arr = np.array([last_tvr[c] for c in code_list], dtype=np.float32)
    pe_arr  = np.array([last_pe[c] for c in code_list], dtype=np.float32)
    pb_arr  = np.array([last_pb[c] for c in code_list], dtype=np.float32)
    cm_arr  = np.array([last_cm[c] for c in code_list], dtype=np.float32)

    def rank_pct(arr):
        valid = ~np.isnan(arr)
        out = np.zeros(n_stocks, dtype=np.float32)
        if valid.sum() >= 2:
            order = np.argsort(np.argsort(arr[valid]))
            out[valid] = order.astype(np.float32) / max(valid.sum() - 1, 1)
        return out

    cross_feat = np.zeros((n_stocks, 7), dtype=np.float32)
    cross_feat[:, 0] = rank_pct(pct_arr)       # pct_chg_rank
    cross_feat[:, 1] = rank_pct(amt_arr)       # amount_rank
    cross_feat[:, 2] = rank_pct(tvr_arr)       # turnover_rate_rank
    cross_feat[:, 3] = pct_arr - idx_pct       # rel_beta
    cross_feat[:, 4] = rank_pct(pe_arr)        # pe_rank
    cross_feat[:, 5] = rank_pct(pb_arr)        # pb_rank
    cross_feat[:, 6] = rank_pct(cm_arr)        # circ_mv_rank

    cross_norm = normalize_cross(cross_feat)   # [N, 7] Z-score across stocks

    # ---- Build final [N, 60, 26] tensor ----
    batch_feat, batch_code = [], []
    for i, ts_code in enumerate(code_list):
        vals_rawtech, vwap_gap = windows[ts_code]
        temporal = np.concatenate([vals_rawtech, vwap_gap[:, None]], axis=1)  # [T, 19]
        temporal = normalize_temporal(temporal)                                # [T, 19]
        cross_tiled = np.tile(cross_norm[i], (WINDOW_SIZE, 1))                # [T, 7]
        feat = np.concatenate([temporal, cross_tiled], axis=-1)               # [T, 26]
        batch_feat.append(feat)
        batch_code.append(ts_code)

    # ------------------------------------------------------------------
    # 6. Model inference
    # ------------------------------------------------------------------
    print(f"[infer] Predicting {len(batch_feat)} stocks ...")
    batch_tensor = torch.from_numpy(np.stack(batch_feat)).to(device)

    with torch.no_grad():
        all_scores = model(batch_tensor).cpu().numpy()

    idx_sorted = np.argsort(all_scores)[::-1]
    top_idx = idx_sorted[:buy_n]
    bot_idx = idx_sorted[-sell_k:]

    # ------------------------------------------------------------------
    # 7. Output
    # ------------------------------------------------------------------
    print(f"\n  BUY  (top {buy_n} stocks for next trading day):")
    for i in top_idx:
        print(f"    {batch_code[i]:12s}  score={all_scores[i]:+.4f}")
    print(f"\n  SELL (bottom {sell_k} stocks):")
    for i in bot_idx:
        print(f"    {batch_code[i]:12s}  score={all_scores[i]:+.4f}")

    if args.dry_run:
        print("\n  [dry-run] Files NOT written.")
    else:
        with open(BUY_PATH, "w") as f:
            f.write("\n".join(batch_code[i] for i in top_idx))
        with open(SELL_PATH, "w") as f:
            f.write("\n".join(batch_code[i] for i in bot_idx))
        print(f"\n  Saved: {os.path.relpath(BUY_PATH, ROOT)}")
        print(f"  Saved: {os.path.relpath(SELL_PATH, ROOT)}")

    # ------------------------------------------------------------------
    # 8. Decision log
    # ------------------------------------------------------------------
    log_exists = os.path.exists(LOG_PATH)
    with open(LOG_PATH, "a") as f:
        if not log_exists:
            f.write(f"{'date':<10} {'pos':>6} {'csi5d':>8} {'trigger':>8} {'buy_n':>5} {'sell_k':>6}\n")
        f.write(f"{latest_date:<10} {position:>5.0%} {csi5d:>+7.2f}% {'YES' if triggered else 'no':>8} {buy_n:>5} {sell_k:>6}\n")
    print(f"  Logged:   {os.path.relpath(LOG_PATH, ROOT)}")

    # ------------------------------------------------------------------
    # 9. Summary
    # ------------------------------------------------------------------
    print(f"\n  {'='*55}")
    print(f"  DATE:     {latest_date}")
    print(f"  CSI300 today: {idx_pct:+.2f}%")
    print(f"  CSI300 5d:    {csi5d:+.2f}%")
    print(f"  POSITION:     {position*100:.0f}%  {'[RISK-OFF]' if triggered else 'FULL'}")
    print(f"  BUY:  {buy_n} stocks -> {os.path.basename(BUY_PATH)}")
    print(f"  SELL: {sell_k} stocks -> {os.path.basename(SELL_PATH)}")
    print(f"  {'='*55}")


if __name__ == "__main__":
    main()
