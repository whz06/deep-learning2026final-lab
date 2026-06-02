"""v6/precompute_backtest.py — Precompute v6 daily scores + 10x10d random backtest.

1. Precomputes v6 spatial model scores for all 2026 trading days
2. Saves to v6/results/daily_scores.parquet for future reuse
3. Runs 10 random 10-day backtests against CSI300 & equal-weight baselines
"""
import os, sys, gc, re, torch, numpy as np, pandas as pd
from scipy.stats import spearmanr

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))  # v6/
ROOT = os.path.dirname(SCRIPT_DIR)  # train_li/

WINDOW_SIZE = 60
RAW_FEATURES = ["open","high","low","close","vol","amount","pct_chg",
                "turnover_rate","volume_ratio","total_mw"]  # typo: total_mv
RAW_FEATURES_FIX = ["open","high","low","close","vol","amount","pct_chg",
                    "turnover_rate","volume_ratio","total_mv"]
TECH_FEATURES = ["macd","macd_signal","rsi","bb_width","bb_pct",
                 "mom_5","mom_20","vol_20"]
LOG_COLS = ["vol", "amount", "total_mv"]
PE_CLIP = (0.1, 500.0)
PB_CLIP = (0.1, 50.0)
WINSOR_P = (1, 99)

CKPT_NAME = "gru_spatial_concat_d32_K5_H128_L1_D0.2_lr0.0003_N1024.pt"
CKPT_PATH = os.path.join(SCRIPT_DIR, "checkpoints", CKPT_NAME)
SCORES_PATH = os.path.join(SCRIPT_DIR, "results", "daily_scores.parquet")
BACKTEST_PATH = os.path.join(SCRIPT_DIR, "results", "backtest_10x10.csv")

TOP_N = 20
SELL_K = 5
START_DATE = "20260203"
END_DATE = "20260529"

# ============================================================
# Inlined model classes (same as trade/infer.py)
# ============================================================
class GRURanker(torch.nn.Module):
    def __init__(self, input_dim, hidden_size=128, num_layers=1, dropout=0.2, bidirectional=False):
        super().__init__()
        self.gru = torch.nn.GRU(input_size=input_dim, hidden_size=hidden_size,
                                num_layers=num_layers, batch_first=True,
                                dropout=dropout if num_layers > 1 else 0.0,
                                bidirectional=bidirectional)
        dir_mult = 2 if bidirectional else 1
        self.head = torch.nn.Sequential(
            torch.nn.Linear(hidden_size * dir_mult, hidden_size // 2),
            torch.nn.ReLU(),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(hidden_size // 2, 1),
        )
    def forward(self, x):
        out, _ = self.gru(x)
        return self.head(out[:, -1, :]).squeeze(-1)

class SparseSpatialAttention(torch.nn.Module):
    def __init__(self, d_model=128, d_proj=32, K=5):
        super().__init__()
        self.query = torch.nn.Linear(d_model, d_proj, bias=False)
        self.key   = torch.nn.Linear(d_model, d_proj, bias=False)
        self.value = torch.nn.Linear(d_model, d_proj, bias=False)
        self.K = K; self.scale = d_proj ** 0.5
    def forward(self, h):
        N = h.size(0)
        q, k, v = self.query(h), self.key(h), self.value(h)
        sim = q @ k.T / self.scale; sim.fill_diagonal_(-float('inf'))
        K_eff = min(self.K, N - 1)
        topk_sim, topk_idx = sim.topk(K_eff, dim=-1)
        attn = torch.nn.functional.softmax(topk_sim, dim=-1)
        return (attn.unsqueeze(1) @ v[topk_idx]).squeeze(1)

class GRURankerSpatialConcat(GRURanker):
    def __init__(self, input_dim, hidden_size=128, num_layers=1, dropout=0.2,
                 bidirectional=False, d_proj=32, K=5):
        super().__init__(input_dim, hidden_size, num_layers, dropout, bidirectional)
        self.spatial = SparseSpatialAttention(d_model=hidden_size, d_proj=d_proj, K=K)
        self.head = torch.nn.Sequential(
            torch.nn.Linear(hidden_size + d_proj, hidden_size // 2),
            torch.nn.ReLU(),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(hidden_size // 2, 1),
        )
    def forward(self, x):
        out, _ = self.gru(x)
        h = out[:, -1, :]; c = self.spatial(h)
        return self.head(torch.cat([h, c], dim=-1)).squeeze(-1)

# ============================================================
# Feature computation (same pipeline as trade/infer.py)
# ============================================================
def add_tech(df):
    c = df["close"].astype(float)
    e12 = c.ewm(span=12, adjust=False).mean(); e26 = c.ewm(span=26, adjust=False).mean()
    df["macd"] = e12 - e26
    df["macd_signal"] = df["macd"].ewm(span=9, adjust=False).mean()
    d = c.diff(); g = d.clip(lower=0); l = (-d).clip(lower=0)
    rs = g.ewm(alpha=1/14, adjust=False).mean() / (l.ewm(alpha=1/14, adjust=False).mean() + 1e-8)
    df["rsi"] = 100 - 100 / (1 + rs)
    m20 = c.rolling(20).mean(); s20 = c.rolling(20).std()
    df["bb_width"] = 2 * s20 / (m20 + 1e-8)
    df["bb_pct"] = (c - (m20 - 2 * s20)) / (4 * s20 + 1e-8)
    df["mom_5"] = c / c.shift(5) - 1
    df["mom_20"] = c / c.shift(20) - 1
    df["vol_20"] = c.pct_change().rolling(20).std()
    return df

def winsorize_2d(arr, p_low=1, p_high=99):
    lo = np.percentile(arr, p_low, axis=0, keepdims=True)
    hi = np.percentile(arr, p_high, axis=0, keepdims=True)
    return np.clip(arr, lo, hi)

def normalize_temporal(arr):
    m, s = arr.mean(axis=0, keepdims=True), arr.std(axis=0, keepdims=True) + 1e-8
    return (arr - m) / s

def normalize_cross(arr):
    m, s = arr.mean(axis=0, keepdims=True), arr.std(axis=0, keepdims=True) + 1e-8
    return (arr - m) / s

def rank_pct(arr, n_stocks):
    valid = ~np.isnan(arr)
    out = np.zeros(n_stocks, dtype=np.float32)
    if valid.sum() >= 2:
        order = np.argsort(np.argsort(arr[valid]))
        out[valid] = order.astype(np.float32) / max(valid.sum() - 1, 1)
    return out

# ============================================================
# Main
# ============================================================
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[precompute] Device: {device}")

    # --- Load model ---
    print(f"[precompute] Loading model: {CKPT_NAME}")
    state = torch.load(CKPT_PATH, map_location=device, weights_only=True)
    d_proj = int(re.search(r"_d(\d+)_", CKPT_NAME).group(1))
    K = int(re.search(r"_K(\d+)_", CKPT_NAME).group(1))
    model = GRURankerSpatialConcat(input_dim=26, hidden_size=128, num_layers=1,
                                    dropout=0.2, d_proj=d_proj, K=K)
    model.load_state_dict(state)
    model.to(device).eval()
    print(f"[precompute] Params: {sum(p.numel() for p in model.parameters()):,}")

    # --- Load data ---
    print("[precompute] Loading parquet ...")
    df = pd.read_parquet(os.path.join(ROOT, "processed", "all_data.parquet"))
    df["trade_date"] = df["trade_date"].astype(str)
    dates_all = sorted(df["trade_date"].unique())

    csi = pd.read_csv(os.path.join(ROOT, "data", "market", "000300.SH.csv"), dtype={"trade_date": str})
    idx_map = dict(zip(csi["trade_date"], csi["pct_chg"].astype(np.float32)))

    # --- Filter to 2026 test period ---
    test_start_idx = dates_all.index(START_DATE) if START_DATE in dates_all else None
    test_end_idx = dates_all.index(END_DATE) if END_DATE in dates_all else None
    if test_start_idx is None:
        print(f"ERROR: {START_DATE} not in dates")
        return

    # Warmup: 300 trading days before first test date
    warmup_cutoff = max(0, test_start_idx - 300)
    df = df[df["trade_date"] >= dates_all[warmup_cutoff]]
    print(f"[precompute] Data range: {df['trade_date'].min()} ~ {df['trade_date'].max()}, "
          f"stocks={df['ts_code'].nunique()}")

    # --- Precompute daily scores ---
    test_dates = [d for d in dates_all if START_DATE <= d <= END_DATE]
    print(f"[precompute] Scoring {len(test_dates)} trading days ({START_DATE} ~ {END_DATE}) ...")

    groups = {ts: sdf.sort_values("trade_date").reset_index(drop=True)
              for ts, sdf in df.groupby("ts_code")}

    records = []
    for di, tgt_date in enumerate(test_dates):
        # Collect per-stock data up to tgt_date
        windows = {}
        last_pct, last_amt, last_tvr = {}, {}, {}
        last_pe, last_pb, last_cm = {}, {}, {}

        for ts_code, sdf in groups.items():
            sdf_cut = sdf[sdf["trade_date"] <= tgt_date]
            if len(sdf_cut) < WINDOW_SIZE:
                continue
            sdf_cut = sdf_cut.ffill().copy()
            sdf_cut = add_tech(sdf_cut)

            vals_raw = sdf_cut[RAW_FEATURES_FIX + TECH_FEATURES].values.astype(np.float32)[-WINDOW_SIZE:]
            for col_name in LOG_COLS:
                if col_name in RAW_FEATURES_FIX:
                    idx = RAW_FEATURES_FIX.index(col_name)
                    vals_raw[:, idx] = np.log1p(np.maximum(vals_raw[:, idx], 0))

            w_close = sdf_cut["close"].values.astype(np.float32)[-WINDOW_SIZE:]
            w_vwap = sdf_cut["vwap"].values.astype(np.float32)[-WINDOW_SIZE:]
            vwap_gap = w_close / np.maximum(w_vwap, 1e-8) - 1

            if np.isnan(vals_raw).any() or np.isnan(vwap_gap).any():
                continue

            windows[ts_code] = (vals_raw, vwap_gap)
            lr = sdf_cut.iloc[-1]
            last_pct[ts_code] = lr["pct_chg"]
            last_amt[ts_code] = lr["amount"]
            last_tvr[ts_code] = lr["turnover_rate"]
            last_pe[ts_code] = lr.get("pe", 0)
            last_pb[ts_code] = lr.get("pb", 0)
            last_cm[ts_code] = lr.get("circ_mv", 0)

        code_list = list(windows.keys())
        N = len(code_list)
        if N < 30:
            continue

        idx_pct = idx_map.get(tgt_date, np.float32(0.0))

        # Cross-sectional arrays
        pct_arr = np.array([last_pct[c] for c in code_list], dtype=np.float32)
        amt_arr = np.array([last_amt[c] for c in code_list], dtype=np.float32)
        tvr_arr = np.array([last_tvr[c] for c in code_list], dtype=np.float32)
        pe_arr  = np.array([last_pe[c] for c in code_list], dtype=np.float32)
        pb_arr  = np.array([last_pb[c] for c in code_list], dtype=np.float32)
        cm_arr  = np.array([last_cm[c] for c in code_list], dtype=np.float32)

        cross_feat = np.zeros((N, 7), dtype=np.float32)
        cross_feat[:, 0] = rank_pct(pct_arr, N)
        cross_feat[:, 1] = rank_pct(amt_arr, N)
        cross_feat[:, 2] = rank_pct(tvr_arr, N)
        cross_feat[:, 3] = pct_arr - idx_pct
        cross_feat[:, 4] = rank_pct(np.clip(pe_arr, PE_CLIP[0], PE_CLIP[1]), N)
        cross_feat[:, 5] = rank_pct(np.clip(pb_arr, PB_CLIP[0], PB_CLIP[1]), N)
        cross_feat[:, 6] = rank_pct(cm_arr, N)
        cross_norm = normalize_cross(cross_feat)

        # Temporal + winsorize
        temporal_all = []
        for ts_code in code_list:
            vr, vg = windows[ts_code]
            temporal_all.append(np.concatenate([vr, vg[:, None]], axis=1))
        t_stack = np.stack(temporal_all, axis=0)  # [N, T, 19]
        Ns, Ts, Fs = t_stack.shape
        t_flat = t_stack.reshape(-1, Fs)
        t_flat = winsorize_2d(t_flat, p_low=WINSOR_P[0], p_high=WINSOR_P[1])
        t_stack = t_flat.reshape(Ns, Ts, Fs)

        # Build batch
        batch_feat = []
        for i in range(N):
            t_norm = normalize_temporal(t_stack[i])
            cross_tiled = np.tile(cross_norm[i], (WINDOW_SIZE, 1))
            batch_feat.append(np.concatenate([t_norm, cross_tiled], axis=-1))

        # Predict
        batch_tensor = torch.from_numpy(np.stack(batch_feat)).float().to(device)
        with torch.no_grad():
            scores = model(batch_tensor).cpu().numpy()

        # Store
        for i, ts_code in enumerate(code_list):
            records.append({"trade_date": tgt_date, "ts_code": ts_code, "score": float(scores[i])})

        torch.cuda.empty_cache()
        if (di + 1) % 10 == 0 or di == 0:
            print(f"  [{di+1}/{len(test_dates)}] {tgt_date}: N={N}, "
                  f"score range [{scores.min():.2f}, {scores.max():.2f}]", flush=True)

    # --- Save scores ---
    scores_df = pd.DataFrame(records)
    os.makedirs(os.path.dirname(SCORES_PATH), exist_ok=True)
    scores_df.to_parquet(SCORES_PATH, index=False)
    print(f"\n[precompute] Saved {len(scores_df):,} scores over {scores_df['trade_date'].nunique()} days → {SCORES_PATH}")

    # ============================================================
    # 10x10d Random Backtest
    # ============================================================
    print(f"\n{'='*60}")
    print(" 10x10d Random Backtest")
    print(f"{'='*60}")

    np.random.seed(42)
    date_list = sorted(scores_df["trade_date"].unique())
    n_possible = len(date_list)
    n_windows = 10
    window_len = 10
    valid_starts = [d for d in date_list
                    if date_list.index(d) < n_possible - window_len - 1]

    starts = sorted(np.random.choice(valid_starts, size=n_windows, replace=False))

    # Build score lookup: dict[date] = {ts_code: score}
    score_lookup = {}
    for _, row in scores_df.iterrows():
        score_lookup.setdefault(row["trade_date"], {})[row["ts_code"]] = row["score"]

    results = []
    for wi, start_date in enumerate(starts):
        start_idx = date_list.index(start_date)
        end_date = date_list[start_idx + window_len - 1]
        end_idx = start_idx + window_len

        # Buy at start_date (scores based on start_date features → next day return)
        buy_date = date_list[start_idx + 1] if start_idx + 1 < len(date_list) else start_date
        buy_scores = score_lookup.get(start_date, {})

        # Sort by score descending, pick top-N
        sorted_items = sorted(buy_scores.items(), key=lambda x: x[1], reverse=True)
        top_codes = [c for c, _ in sorted_items[:TOP_N]]

        # Compute cumulative returns for portfolio
        group_data = groups  # pre-loaded stock data

        portfolio_rets = []
        csi_rets = []
        all_rets = []

        for t in range(window_len):
            hold_date = date_list[start_idx + 1 + t]  # day after buy
            next_date = date_list[start_idx + 1 + t + 1] if start_idx + 1 + t + 1 < len(date_list) else hold_date

            # Portfolio return: average pct_chg of top-N stocks from hold_date to next_date
            stock_rets = []
            for ts_code in top_codes:
                if ts_code not in group_data:
                    continue
                sdf = group_data[ts_code]
                row_hold = sdf[sdf["trade_date"] == hold_date]
                row_next = sdf[sdf["trade_date"] == next_date]
                if len(row_hold) == 0 or len(row_next) == 0:
                    continue
                ret = float(row_next["pct_chg"].iloc[0])
                stock_rets.append(ret)

            # CSI300 return
            csi_ret = idx_map.get(hold_date, 0.0)

            # Equal-weight all stocks
            all_valid = []
            for ts_code in group_data:
                sdf = group_data[ts_code]
                row_hold = sdf[sdf["trade_date"] == hold_date]
                row_next = sdf[sdf["trade_date"] == next_date]
                if len(row_hold) > 0 and len(row_next) > 0:
                    all_valid.append(float(row_next["pct_chg"].iloc[0]))
            ew_ret = np.mean(all_valid) if all_valid else 0.0

            if stock_rets:
                portfolio_rets.append(np.mean(stock_rets))
            csi_rets.append(csi_ret)
            all_rets.append(ew_ret)

        cum_port = np.sum(portfolio_rets) if portfolio_rets else 0.0
        cum_csi = np.sum(csi_rets) if csi_rets else 0.0
        cum_ew = np.sum(all_rets) if all_rets else 0.0
        win = cum_port > cum_csi

        results.append({
            "window": wi + 1,
            "start": start_date,
            "end": end_date,
            "cum_return_pct": round(cum_port, 4),
            "csi300_cum_pct": round(cum_csi, 4),
            "ew_cum_pct": round(cum_ew, 4),
            "excess_vs_csi": round(cum_port - cum_csi, 4),
            "excess_vs_ew": round(cum_port - cum_ew, 4),
            "beat_csi": win,
        })
        print(f"  Win {wi+1}: {start_date}~{end_date} | portfolio={cum_port:+.3f}% csi={cum_csi:+.3f}% "
              f"ew={cum_ew:+.3f}% | {'WIN' if win else 'LOSS'}")

    # Summary
    cum_rets = [r["cum_return_pct"] for r in results]
    csi_rets_sum = [r["csi300_cum_pct"] for r in results]
    wins = sum(r["beat_csi"] for r in results)
    avg_excess = np.mean([r["excess_vs_csi"] for r in results])

    print(f"\n{'='*60}")
    print(f" Summary")
    print(f"{'='*60}")
    print(f"  Wins: {wins}/{n_windows} ({wins/n_windows*100:.0f}%)")
    print(f"  Mean return:     {np.mean(cum_rets):+.3f}%")
    print(f"  Mean CSI300:     {np.mean(csi_rets_sum):+.3f}%")
    print(f"  Mean excess:     {avg_excess:+.3f}%")
    print(f"  Std return:      {np.std(cum_rets):.3f}%")

    df_res = pd.DataFrame(results)
    df_res.to_csv(BACKTEST_PATH, index=False)
    print(f"\n  Results saved → {BACKTEST_PATH}")

    print(f"\n[precompute] All done.")
    torch.cuda.empty_cache()

if __name__ == "__main__":
    main()
