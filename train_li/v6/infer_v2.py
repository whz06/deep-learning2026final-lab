"""v6/infer_v2.py — v2-style inference with original 22-dim + old normalization.

Uses v2 GRU checkpoint: gru_gru_hidden_size=128_num_layers=1_dropout=0.2_lr=0.0003.pt
Feature pipeline matches v2 training (22-dim, per-window Z-score, no v5 preprocessing).
Strategy B applied: csi5d < -1% -> 80% position.
"""
import os, sys, re, argparse, numpy as np, pandas as pd, torch, torch.nn as nn

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(SCRIPT_DIR)

PARQUET_PATH = os.path.join(ROOT, "processed", "all_data.parquet")
INDEX_PATH   = os.path.join(ROOT, "data", "market", "000300.SH.csv")
CKPT_PATH    = os.path.join(ROOT, "v2", "checkpoints", "gru_gru_hidden_size=128_num_layers=1_dropout=0.2_lr=0.0003.pt")
BUY_PATH     = os.path.join(ROOT, "trade", "buy_list.txt")
SELL_PATH    = os.path.join(ROOT, "trade", "sell_list.txt")
LOG_PATH     = os.path.join(ROOT, "trade", "decision.log")

W = 60
WARMUP = 200
RAW = ["open","high","low","close","vol","amount","pct_chg","turnover_rate","volume_ratio","total_mv"]
TECH = ["macd","macd_signal","rsi","bb_width","bb_pct","mom_5","mom_20","vol_20"]
INPUT_DIM = 22
THRESH = -1.0  # Strategy B
MIN_POS = 0.80

class GRURanker(nn.Module):
    def __init__(self, input_dim, hidden_size=128, num_layers=1, dropout=0.2, bidirectional=False):
        super().__init__()
        self.gru = nn.GRU(input_size=input_dim, hidden_size=hidden_size,
                          num_layers=num_layers, batch_first=True,
                          dropout=dropout if num_layers > 1 else 0.0, bidirectional=bidirectional)
        d = hidden_size * (2 if bidirectional else 1)
        self.head = nn.Sequential(nn.Linear(d, hidden_size//2), nn.ReLU(), nn.Dropout(dropout), nn.Linear(hidden_size//2, 1))
    def forward(self, x):
        out, _ = self.gru(x)
        return self.head(out[:, -1, :]).squeeze(-1)

def add_tech(df):
    c = df["close"].astype(float)
    e12, e26 = c.ewm(span=12, adjust=False).mean(), c.ewm(span=26, adjust=False).mean()
    df["macd"] = e12 - e26
    df["macd_signal"] = df["macd"].ewm(span=9, adjust=False).mean()
    d = c.diff(); g = d.clip(lower=0); l = (-d).clip(lower=0)
    rs = g.ewm(alpha=1/14, adjust=False).mean() / (l.ewm(alpha=1/14, adjust=False).mean() + 1e-8)
    df["rsi"] = 100 - 100 / (1 + rs)
    m20, s20 = c.rolling(20).mean(), c.rolling(20).std()
    df["bb_width"] = 2 * s20 / (m20 + 1e-8)
    df["bb_pct"] = (c - (m20 - 2 * s20)) / (4 * s20 + 1e-8)
    df["mom_5"] = c / c.shift(5) - 1
    df["mom_20"] = c / c.shift(20) - 1
    df["vol_20"] = c.pct_change().rolling(20).std()
    return df

def compute_csi5d(csi_series, target_date):
    csi_series = csi_series.sort_index()
    idx = csi_series.index.tolist()
    if target_date not in idx:
        prior = [d for d in idx if d <= target_date]
        if not prior: return 0.0, 0.0
        target_date = prior[-1]
    pos = idx.index(target_date)
    start = max(0, pos - 4)
    vals = csi_series.iloc[start:pos+1].values
    csi5d = float(np.sum(vals))
    return csi5d, float(csi_series.loc[target_date])

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--date", type=str, default="20260601")
    parser.add_argument("--top-n", type=int, default=20)
    parser.add_argument("--bottom-k", type=int, default=5)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--no-strategy", action="store_true")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[v2-infer] Device: {device}")

    # Load model
    print("[v2-infer] Loading v2 GRU ...")
    state = torch.load(CKPT_PATH, map_location=device, weights_only=True)
    model = GRURanker(INPUT_DIM, 128, 1, 0.2).to(device).eval()
    model.load_state_dict(state)
    print(f"[v2-infer] v2 GRU loaded")

    # Load data
    print("[v2-infer] Loading data ...")
    df = pd.read_parquet(PARQUET_PATH)
    df["trade_date"] = df["trade_date"].astype(str)
    dates_all = sorted(df["trade_date"].unique())
    latest_date = args.date if args.date else dates_all[-1]
    if latest_date not in dates_all:
        print(f"ERROR: {latest_date} not in data")
        sys.exit(1)
    date_idx = dates_all.index(latest_date)
    cutoff_idx = max(0, date_idx - WARMUP)
    df = df[df["trade_date"].isin(dates_all[cutoff_idx:])]
    print(f"[v2-infer] Feature end: {latest_date} (history: {date_idx - cutoff_idx} days)")

    # CSI300
    csi = pd.read_csv(INDEX_PATH, dtype={"trade_date": str})
    idx_map = dict(zip(csi["trade_date"], csi["pct_chg"].astype(float)))
    idx_pct = idx_map.get(latest_date, 0.0)
    csi_series = pd.Series(idx_map.values(), index=list(idx_map.keys()))
    csi5d, _ = compute_csi5d(csi_series, latest_date)

    # Strategy B
    position = MIN_POS if csi5d < THRESH and not args.no_strategy else 1.0
    buy_n = max(1, int(np.ceil(args.top_n * position)))
    sell_k = max(1, int(np.floor(args.bottom_k * position)))
    triggered = csi5d < THRESH and not args.no_strategy

    print(f"\n{'='*55}")
    print(f"  STRATEGY B: CSI5d={csi5d:+.2f}% {'< -1% -> 80%' if triggered else '-> FULL'}")
    print(f"  Position: {position*100:.0f}%  buy_n={buy_n}  sell_k={sell_k}")
    print(f"{'='*55}")

    # Feature computation (v2 style: per-window Z-score)
    print("[v2-infer] Computing features ...")
    grouped = df.groupby("ts_code")

    # Pre-compute stock series
    series = {}
    for ts, sdf in grouped:
        sdf = sdf.sort_values("trade_date")
        sdf = sdf[sdf["trade_date"] <= latest_date].copy()
        if len(sdf) < W + 1: continue
        sdf = sdf.ffill()
        sdf = add_tech(sdf)
        series[ts] = sdf

    # Build features
    feats, codes = [], []
    for ts, sdf in series.items():
        sw = sdf.iloc[-W-1:]
        rv = sw[RAW + TECH].values.astype(np.float32)
        if np.isnan(rv).any(): continue
        feats.append(rv[-W:])
        codes.append(ts)

    if not feats:
        print("[v2-infer] ERROR: No valid stocks")
        sys.exit(1)

    fa = np.stack(feats, 0)  # [N, T, 18]
    n = len(feats)
    pv, av, tv = fa[:, -1, 6], fa[:, -1, 5], fa[:, -1, 7]  # pct_chg, amount, turnover at t=-1
    pr = np.argsort(np.argsort(pv)).astype(np.float32) / max(n - 1, 1)
    ar = np.argsort(np.argsort(av)).astype(np.float32) / max(n - 1, 1)
    tr = np.argsort(np.argsort(tv)).astype(np.float32) / max(n - 1, 1)
    ip = idx_map.get(latest_date, 0.0)
    rb = np.full(n, pv - ip, dtype=np.float32)

    cr = np.stack([np.tile(pr[:, None], (1, W)), np.tile(ar[:, None], (1, W)),
                    np.tile(tr[:, None], (1, W)), np.tile(rb[:, None], (1, W))], -1)
    full = np.concatenate([fa, cr], -1)

    # v2 normalization: per-window Z-score
    m, s = full.mean(axis=0, keepdims=True), full.std(axis=0, keepdims=True) + 1e-8
    full_norm = (full - m) / s

    print(f"[v2-infer] Valid stocks: {n}")
    batch = torch.from_numpy(full_norm).float().to(device)

    with torch.no_grad():
        scores = model(batch).cpu().numpy()

    idx_sorted = np.argsort(scores)[::-1]
    top_idx = idx_sorted[:buy_n]
    bot_idx = idx_sorted[-sell_k:]

    print(f"\n  BUY (top {buy_n} stocks):")
    for i in top_idx:
        print(f"    {codes[i]:12s}  score={scores[i]:+.4f}")
    print(f"\n  SELL (bottom {sell_k} stocks):")
    for i in bot_idx:
        print(f"    {codes[i]:12s}  score={scores[i]:+.4f}")

    if args.dry_run:
        print("\n  [dry-run] Files NOT written.")
    else:
        with open(BUY_PATH, "w") as f:
            f.write("\n".join(codes[i] for i in top_idx))
        with open(SELL_PATH, "w") as f:
            f.write("\n".join(codes[i] for i in bot_idx))

    log_exists = os.path.exists(LOG_PATH)
    with open(LOG_PATH, "a") as f:
        if not log_exists:
            f.write(f"{'date':<10} {'pos':>6} {'csi5d':>8} {'trigger':>8} {'buy_n':>5} {'sell_k':>6}\n")
        f.write(f"{latest_date:<10} {position:>5.0%} {csi5d:>+7.2f}% {'YES' if triggered else 'no':>8} {buy_n:>5} {sell_k:>6}\n")

    print(f"\n  DATE: {latest_date}  CSI5d={csi5d:+.2f}%  POS={position*100:.0f}%")
    print(f"  BUY {buy_n}: {BUY_PATH}")
    print(f"  SELL {sell_k}: {SELL_PATH}")

if __name__ == "__main__":
    main()
