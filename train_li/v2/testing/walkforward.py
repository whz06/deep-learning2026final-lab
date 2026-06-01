"""
v2/testing/walkforward.py — 10-segment walk-forward test on 2026 data.
Supports: single GRU / heterogeneous ensemble / seed ensemble.

Usage:
  python walkforward.py                      # single GRU
  python walkforward.py --mode ensemble      # GRU+TF weighted fusion
  python walkforward.py --mode seeds         # GRU × 5 seeds average
"""
import os, sys, re, gc, argparse
import numpy as np
import pandas as pd
import torch

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
V2_DIR = os.path.dirname(SCRIPT_DIR)
ROOT = os.path.dirname(V2_DIR)
PARQUET_PATH = os.path.join(ROOT, "processed", "all_data.parquet")
INDEX_PATH   = os.path.join(ROOT, "data", "market", "000300.SH.csv")
CKPT_DIR = os.path.join(V2_DIR, "checkpoints")
SCORE_DIR = os.path.join(V2_DIR, "scores")

sys.path.insert(0, V2_DIR)
from models.gru import GRURanker
from models.transformer import TransformerRanker
from models.mlp import MLPRanker

# ── Constants ──
WINDOW_SIZE = 60
RAW_FEATURES = ["open","high","low","close","vol","amount","pct_chg",
                "turnover_rate","volume_ratio","total_mv"]
TECH_FEATURES = ["macd","macd_signal","rsi","bb_width","bb_pct",
                 "mom_5","mom_20","vol_20"]
TOP_N, REBALANCE_K = 20, 2
TRADE_COST = 0.0013
LIMIT_UP_THRESHOLD = 9.9
TEST_START, TEST_END = "20260201", "20260531"
SEGMENT_DAYS, STEP_DAYS = 10, 7

# Best ckpts
GRU_CKPT = "gru_gru_hidden_size=128_num_layers=1_dropout=0.2_lr=0.0003.pt"
TF_CKPT  = "tf_tf_d_model=96_n_heads=4_n_temporal_layers=2_n_spatial_layers=1_dropout=0.1_lr=0.0003.pt"
SEEDS = [42, 123, 456, 789, 1024]


# ── Tech indicators ──
def add_tech_indicators(df):
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

def _normalize(arr):
    m, s = arr.mean(axis=0, keepdims=True), arr.std(axis=0, keepdims=True) + 1e-8
    return (arr - m) / s

def _rank(arr):
    return np.argsort(np.argsort(arr)).astype(np.float32) / max(len(arr) - 1, 1)


# ── Model loading ──

def load_models(mode, device):
    """Returns list of (model, weight, name) tuples."""
    models = []
    if mode == "single":
        m = GRURanker(input_dim=22, hidden_size=128, num_layers=1, dropout=0.2)
        m.load_state_dict(torch.load(os.path.join(CKPT_DIR, GRU_CKPT), map_location=device, weights_only=True))
        m.to(device).eval()
        models.append((m, 1.0, "GRU"))
        print(f"[model] Single GRU")

    elif mode == "ensemble":
        # GRU (weight=0.6)
        m_gru = GRURanker(input_dim=22, hidden_size=128, num_layers=1, dropout=0.2)
        m_gru.load_state_dict(torch.load(os.path.join(CKPT_DIR, GRU_CKPT), map_location=device, weights_only=True))
        m_gru.to(device).eval()
        models.append((m_gru, 0.6, "GRU"))

        # TF (weight=0.4)
        m_tf = TransformerRanker(input_dim=22, d_model=96, n_heads=4, n_temporal_layers=2, n_spatial_layers=1, dropout=0.1)
        m_tf.load_state_dict(torch.load(os.path.join(CKPT_DIR, TF_CKPT), map_location=device, weights_only=True))
        m_tf.to(device).eval()
        models.append((m_tf, 0.4, "TF"))
        print(f"[model] Ensemble: GRU×0.6 + TF×0.4")

    elif mode == "seeds":
        for seed in SEEDS:
            ckpt = os.path.join(CKPT_DIR, f"gru_seed{seed}.pt")
            m = GRURanker(input_dim=22, hidden_size=128, num_layers=1, dropout=0.2)
            m.load_state_dict(torch.load(ckpt, map_location=device, weights_only=True))
            m.to(device).eval()
            models.append((m, 1.0 / len(SEEDS), f"GRU_s{seed}"))
        print(f"[model] Seed ensemble: {len(SEEDS)} GRU seeds")

    elif mode == "tf":
        m = TransformerRanker(input_dim=22, d_model=96, n_heads=4, n_temporal_layers=2, n_spatial_layers=1, dropout=0.1)
        m.load_state_dict(torch.load(os.path.join(CKPT_DIR, TF_CKPT), map_location=device, weights_only=True))
        m.to(device).eval()
        models.append((m, 1.0, "TF"))
        print(f"[model] TF single")
    return models


def ensemble_predict(feat_tensor, models, device):
    """Weighted average of scores from multiple models."""
    total = np.zeros(feat_tensor.shape[0], dtype=np.float32)
    with torch.no_grad():
        for model, weight, _ in models:
            s = model(feat_tensor).cpu().numpy()
            total += weight * s
            torch.cuda.empty_cache()
    return total


# ── Feature building (same as before) ──

def build_day_features(date, stock_series, idx_pct_map):
    feats, codes = [], []
    for ts_code, sdf in stock_series.items():
        sdf_date = sdf[sdf["trade_date"] <= date]
        if len(sdf_date) < WINDOW_SIZE + 1: continue
        sdf_window = sdf_date.iloc[-WINDOW_SIZE-1:]
        raw_vals = sdf_window[RAW_FEATURES + TECH_FEATURES].values.astype(np.float32)
        if np.isnan(raw_vals).any(): continue
        feats.append(raw_vals[-WINDOW_SIZE:]); codes.append(ts_code)

    if len(feats) == 0: return None, None
    feat_arr = np.stack(feats, axis=0); n = len(codes)

    pct_vals  = feat_arr[:, -1, RAW_FEATURES.index("pct_chg")]
    amt_vals  = feat_arr[:, -1, RAW_FEATURES.index("amount")]
    tnvr_vals = feat_arr[:, -1, RAW_FEATURES.index("turnover_rate")]
    idx_pct   = idx_pct_map.get(date, 0.0)

    cross = np.stack([
        np.tile(_rank(pct_vals)[:, None],  (1, WINDOW_SIZE)),
        np.tile(_rank(amt_vals)[:, None],  (1, WINDOW_SIZE)),
        np.tile(_rank(tnvr_vals)[:, None], (1, WINDOW_SIZE)),
        np.tile(np.full(n, pct_vals - idx_pct, dtype=np.float32)[:, None], (1, WINDOW_SIZE)),
    ], axis=-1)
    full = np.concatenate([feat_arr, cross], axis=-1)
    for j in range(n): full[j] = _normalize(full[j])
    return full, codes


# ── Run one segment ──

def run_segment(seg_dates, models, stock_series, idx_pct_map, device):
    holdings, daily_rets, turnover = [], [], 0
    for pi in range(SEGMENT_DAYS):
        pdate, ndate = seg_dates[pi], seg_dates[pi + 1]
        features, codes = build_day_features(pdate, stock_series, idx_pct_map)
        if features is None: continue
        feat_t = torch.from_numpy(features).to(device)
        scores = ensemble_predict(feat_t, models, device)

        # Next-day returns
        valid_idx, labels_l = [], []
        for ci, c in enumerate(codes):
            r = stock_series[c][stock_series[c]["trade_date"] == ndate]
            if len(r) > 0: valid_idx.append(ci); labels_l.append(r["pct_chg"].values[0])
        if not valid_idx: continue
        cv, sv, lv = [codes[i] for i in valid_idx], scores[valid_idx], np.array(labels_l, dtype=np.float32)

        limit_up = features[valid_idx, -1, RAW_FEATURES.index("pct_chg")] >= LIMIT_UP_THRESHOLD

        if not holdings:
            elig = np.where(~limit_up)[0]
            if len(elig) >= TOP_N:
                holdings = [cv[i] for i in elig[np.argsort(sv[elig])[-TOP_N:]]]
        else:
            hs = {c: sv[cv.index(c)] for c in holdings if c in cv}
            if len(hs) >= REBALANCE_K:
                for c in sorted(hs, key=hs.get)[:REBALANCE_K]: holdings.remove(c); turnover += 1
            ch = set(holdings)
            cand = [(i, s) for i, (c, s) in enumerate(zip(cv, sv)) if c not in ch and not limit_up[i]]
            cand.sort(key=lambda x: x[1], reverse=True)
            for i, _ in cand[:REBALANCE_K]: holdings.append(cv[i]); turnover += 1
            holdings = holdings[-TOP_N:]

        hr = [lv[cv.index(c)] for c in holdings if c in cv]
        if hr:
            dr = float(np.mean(hr))
            if pi > 0 and turnover > 0: dr -= turnover * TRADE_COST / max(len(holdings), 1)
            daily_rets.append(dr)
        turnover = 0

    if not daily_rets: return None
    dr = np.array(daily_rets)
    cum_ret = float(np.prod(1 + dr / 100) - 1) * 100
    max_dd = float(np.min(np.cumprod(1 + dr / 100) / np.maximum.accumulate(np.cumprod(1 + dr / 100)) - 1)) * 100
    return {"cum_ret": cum_ret, "max_dd": max_dd, "daily": dr}


# ── Main ──

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", default="single", choices=["single","ensemble","seeds","tf"])
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[device] {device} | mode={args.mode}")

    models = load_models(args.mode, device)

    print("[data] Loading all_data.parquet ...")
    df = pd.read_parquet(PARQUET_PATH)
    df["trade_date"] = df["trade_date"].astype(str)
    all_raw_dates = sorted(df["trade_date"].unique())
    test_candidates = [d for d in all_raw_dates if d >= TEST_START]
    if not test_candidates: print("ERROR: no test dates"); return
    first_test_date = test_candidates[0]
    lb_idx = max(0, all_raw_dates.index(first_test_date) - WINDOW_SIZE - 1 - 20)
    df = df[df["trade_date"] >= all_raw_dates[lb_idx]]
    print(f"[data] {len(df)} rows, {df['trade_date'].nunique()} dates, {df['ts_code'].nunique()} stocks")

    idx_df = pd.read_csv(INDEX_PATH, dtype={"trade_date": str})
    idx_pct_map = dict(zip(idx_df["trade_date"], idx_df["pct_chg"].astype(float)))

    print("[feat] Computing technical indicators ...")
    stocks = sorted(df["ts_code"].unique())
    stock_series = {}
    for i, ts_code in enumerate(stocks):
        sdf = df[df["ts_code"] == ts_code].sort_values("trade_date").reset_index(drop=True)
        stock_series[ts_code] = add_tech_indicators(sdf)
        if (i + 1) % 1000 == 0: print(f"  ... {i+1}/{len(stocks)}")

    test_dates = [d for d in all_raw_dates if TEST_START <= d <= TEST_END]
    print(f"[dates] {len(test_dates)} days ({test_dates[0]}~{test_dates[-1]})")

    segments = []
    i = 0
    while i + SEGMENT_DAYS + 1 <= len(test_dates):
        segments.append(test_dates[i:i + SEGMENT_DAYS + 1])
        i += STEP_DAYS
    print(f"[segments] {len(segments)} segments\n")

    all_results = []
    for si, seg in enumerate(segments):
        res = run_segment(seg, models, stock_series, idx_pct_map, device)
        if res is None: continue
        csi_daily = [idx_pct_map.get(seg[j + 1], 0.0) for j in range(SEGMENT_DAYS)]
        csi_cum = float(np.prod(1 + np.array(csi_daily) / 100) - 1) * 100
        excess = res["cum_ret"] - csi_cum
        w = "W" if res["cum_ret"] > 0 else "L"
        all_results.append({**res, "csi_cum": csi_cum, "excess": excess, "win": w,
                            "start": seg[0], "end": seg[-1], "seg": si + 1})
        print(f"  Seg {si+1:2d} | {seg[0]}→{seg[-1]} | "
              f"Model:{res['cum_ret']:+6.2f}% | CSI300:{csi_cum:+6.2f}% | "
              f"Excess:{excess:+6.2f}% | MaxDD:{res['max_dd']:+5.2f}% | {w}")
        gc.collect(); torch.cuda.empty_cache()

    if not all_results: print("No valid segments"); return
    gru_rets = np.array([r["cum_ret"] for r in all_results])
    csi_rets = np.array([r["csi_cum"] for r in all_results])
    ex_rets  = np.array([r["excess"] for r in all_results])
    wins = sum(1 for r in all_results if r["win"] == "W")

    print(f"\n{'='*80}")
    print(f"  Walk-Forward: {args.mode.upper()} vs CSI300  ({len(all_results)} segments)")
    print(f"{'='*80}")
    print(f"{'Seg':>4}  {'Start':<12} {'End':<12} {'Model':>8} {'CSI300':>8} {'Excess':>8} {'MaxDD':>7} Win")
    print(f"{'-'*80}")
    for r in all_results:
        print(f"{r['seg']:>4}  {r['start']:<12} {r['end']:<12} "
              f"{r['cum_ret']:>7.2f}% {r['csi_cum']:>7.2f}% {r['excess']:>7.2f}% {r['max_dd']:>6.2f}%  {r['win']}")
    print(f"{'-'*80}")
    print(f"  Mean: Model={gru_rets.mean():+.2f}%  CSI300={csi_rets.mean():+.2f}%  Excess={ex_rets.mean():+.2f}%")
    print(f"  Std:  Model={gru_rets.std():.2f}%  CSI300={csi_rets.std():.2f}%  Excess={ex_rets.std():.2f}%")
    print(f"  Max:  Model={gru_rets.max():+.2f}%  CSI300={csi_rets.max():+.2f}%")
    print(f"  Min:  Model={gru_rets.min():+.2f}%  CSI300={csi_rets.min():+.2f}%")
    print(f"  Win/Loss: {wins}/{len(all_results)-wins}")
    print(f"{'='*80}")

    for m in models: del m[0]
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
