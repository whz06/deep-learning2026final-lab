"""v4/eval_t30.py — Compare T=30 vs T=60 GRU on 2026 test period.

Evaluates both models on the same test dates, computing:
  - Daily Rank IC for each model
  - Daily score rank correlation (do T=30 and T=60 agree?)
  - IC correlation between the two models

Output: v4/results/eval_t30.json

Key metric: score_rank_correlation_mean
  > 0.9 → skip ensemble (no diversity gain)
  < 0.7 → ensemble with equal weights (genuine diversity)

Usage:
  python v4/eval_t30.py              # after training T=30
  python v4/eval_t30.py --device cpu  # fallback
"""

import os, sys, json, argparse
import numpy as np
import pandas as pd
import torch
from scipy.stats import spearmanr

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(SCRIPT_DIR)
V2_DIR = os.path.join(ROOT, "v2")
sys.path.insert(0, V2_DIR)
from models.gru import GRURanker

PARQUET_PATH = os.path.join(ROOT, "processed", "all_data.parquet")
INDEX_PATH = os.path.join(ROOT, "data", "market", "000300.SH.csv")
CKPT_T30 = os.path.join(SCRIPT_DIR, "checkpoints", "gru_t30_h128_l1_d0.2.pt")
CKPT_T60 = os.path.join(V2_DIR, "checkpoints",
                        "gru_gru_hidden_size=128_num_layers=1_dropout=0.2_lr=0.0003.pt")
OUT_PATH = os.path.join(SCRIPT_DIR, "results", "eval_t30.json")

RAW = ["open", "high", "low", "close", "vol", "amount",
       "pct_chg", "turnover_rate", "volume_ratio", "total_mv"]
TECH = ["macd", "macd_signal", "rsi", "bb_width", "bb_pct", "mom_5", "mom_20", "vol_20"]


def add_tech(df):
    c = df["close"].astype(float)
    e12 = c.ewm(span=12, adjust=False).mean()
    e26 = c.ewm(span=26, adjust=False).mean()
    df["macd"] = e12 - e26
    df["macd_signal"] = df["macd"].ewm(span=9, adjust=False).mean()
    d = c.diff()
    g = d.clip(lower=0)
    l = (-d).clip(lower=0)
    rs = g.ewm(alpha=1 / 14, adjust=False).mean() / (l.ewm(alpha=1 / 14, adjust=False).mean() + 1e-8)
    df["rsi"] = 100 - 100 / (1 + rs)
    m20 = c.rolling(20).mean()
    s20 = c.rolling(20).std()
    df["bb_width"] = 2 * s20 / (m20 + 1e-8)
    df["bb_pct"] = (c - (m20 - 2 * s20)) / (4 * s20 + 1e-8)
    df["mom_5"] = c / c.shift(5) - 1
    df["mom_20"] = c / c.shift(20) - 1
    df["vol_20"] = c.pct_change().rolling(20).std()
    return df


def build(date, series, idx_map, W):
    feats, codes = [], []
    for ts, sdf in series.items():
        sd = sdf[sdf["trade_date"] <= date]
        if len(sd) < W + 1:
            continue
        sw = sd.iloc[-W - 1:]
        rv = sw[RAW + TECH].values.astype(np.float32)
        if np.isnan(rv).any():
            continue
        feats.append(rv[-W:])
        codes.append(ts)
    if not feats:
        return None, None

    fa = np.stack(feats, axis=0)
    n = len(feats)
    pv = fa[:, -1, 6]
    av = fa[:, -1, 5]
    tv = fa[:, -1, 7]
    pr = np.argsort(np.argsort(pv)).astype(np.float32) / max(n - 1, 1)
    ar = np.argsort(np.argsort(av)).astype(np.float32) / max(n - 1, 1)
    tr = np.argsort(np.argsort(tv)).astype(np.float32) / max(n - 1, 1)
    ip = idx_map.get(date, 0)
    rb = np.full(n, pv - ip, dtype=np.float32)
    cr = np.stack([
        np.tile(pr[:, None], (1, W)),
        np.tile(ar[:, None], (1, W)),
        np.tile(tr[:, None], (1, W)),
        np.tile(rb[:, None], (1, W)),
    ], axis=-1)
    full = np.concatenate([fa, cr], axis=-1)
    m = full.mean(axis=0, keepdims=True)
    s = full.std(axis=0, keepdims=True) + 1e-8
    return (full - m) / s, codes


def compute_ic(scores, codes, next_date, series):
    sl, lb = [], []
    for ci, cc in enumerate(codes):
        r = series[cc][series[cc]["trade_date"] == next_date]
        if len(r) > 0:
            sl.append(scores[ci])
            lb.append(r["pct_chg"].values[0])
    if len(sl) < 30:
        return None
    return spearmanr(sl, lb).correlation


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--n-stocks", type=int, default=600)
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"[eval_t30] device={device}")

    # Load models
    if not os.path.exists(CKPT_T30):
        print(f"[eval_t30] ERROR: T=30 checkpoint not found at {CKPT_T30}")
        print("  Run: python v4/train_t30.py first")
        sys.exit(1)
    if not os.path.exists(CKPT_T60):
        print(f"[eval_t30] ERROR: T=60 checkpoint not found at {CKPT_T60}")
        sys.exit(1)

    m30 = GRURanker(22, 128, 1, 0.2)
    m30.load_state_dict(torch.load(CKPT_T30, map_location=device, weights_only=True))
    m30.to(device).eval()

    m60 = GRURanker(22, 128, 1, 0.2)
    m60.load_state_dict(torch.load(CKPT_T60, map_location=device, weights_only=True))
    m60.to(device).eval()
    print("[eval_t30] Both models loaded")

    # Load data
    df = pd.read_parquet(PARQUET_PATH)
    df["trade_date"] = df["trade_date"].astype(str)
    dates_all = sorted(df["trade_date"].unique())
    test_start_idx = [i for i, d in enumerate(dates_all) if d >= "20260201"][0]
    lb = max(0, test_start_idx - 60 - 1 - 90)
    df = df[df["trade_date"] >= dates_all[lb]]
    print(f"[eval_t30] Data rows: {len(df):,}")

    csi = pd.read_csv(INDEX_PATH, dtype={"trade_date": str})
    idx_map = dict(zip(csi["trade_date"], csi["pct_chg"]))

    stocks = sorted(df["ts_code"].unique())
    np.random.seed(42)
    stocks = sorted(np.random.choice(stocks, min(args.n_stocks, len(stocks)), replace=False))
    print(f"[eval_t30] {len(stocks)} stocks sampled")

    series = {ts: add_tech(df[df["ts_code"] == ts].sort_values("trade_date")
                .reset_index(drop=True)) for ts in stocks}

    test_dates = [d for d in dates_all if "20260201" <= d <= "20260531"]
    print(f"[eval_t30] Test dates: {len(test_dates)} ({test_dates[0]} -> {test_dates[-1]})")

    # Accumulate daily results
    ic30_list, ic60_list = [], []
    n30_list, n60_list = [], []
    rank_corrs = []

    for di, date in enumerate(test_dates):
        if di == len(test_dates) - 1:
            break
        nd = test_dates[di + 1]

        # T=30
        f30, c30 = build(date, series, idx_map, 30)
        if f30 is not None:
            t30 = torch.from_numpy(f30).to(device)
            s30 = m30(t30).detach().cpu().numpy()
            ic30 = compute_ic(s30, c30, nd, series)
            ic30_list.append(ic30 if ic30 is not None else float("nan"))
            n30_list.append(len(c30) if f30 is not None else 0)
        else:
            ic30_list.append(float("nan"))
            n30_list.append(0)

        # T=60
        f60, c60 = build(date, series, idx_map, 60)
        if f60 is not None:
            t60 = torch.from_numpy(f60).to(device)
            s60 = m60(t60).detach().cpu().numpy()
            ic60 = compute_ic(s60, c60, nd, series)
            ic60_list.append(ic60 if ic60 is not None else float("nan"))
            n60_list.append(len(c60))
        else:
            ic60_list.append(float("nan"))
            n60_list.append(0)

        # Score rank correlation on common stocks
        if f30 is not None and f60 is not None:
            common = list(set(c30) & set(c60))
            if len(common) >= 30:
                i30 = [c30.index(c) for c in common]
                i60 = [c60.index(c) for c in common]
                sr = spearmanr(s30[i30], s60[i60]).correlation
                rank_corrs.append(sr)

        torch.cuda.empty_cache()

        if (di + 1) % 20 == 0:
            print(f"  [{di + 1}/{len(test_dates) - 1}] "
                  f"IC30={np.nanmean(ic30_list):+.4f}  "
                  f"IC60={np.nanmean(ic60_list):+.4f}  "
                  f"rank_corr={np.mean(rank_corrs[-20:]):.3f}"
                  if rank_corrs else f"  [{di + 1}/{len(test_dates) - 1}]",
                  flush=True)

    # --- Aggregate ---
    ic30_valid = [v for v in ic30_list if not np.isnan(v)]
    ic60_valid = [v for v in ic60_list if not np.isnan(v)]

    # Pairwise IC correlation (only on days where both are valid)
    ic_pairs = [(a, b) for a, b in zip(ic30_list, ic60_list)
                if not np.isnan(a) and not np.isnan(b)]
    if len(ic_pairs) >= 5:
        ic30_paired, ic60_paired = zip(*ic_pairs)
        ic_corr = float(np.corrcoef(ic30_paired, ic60_paired)[0, 1])
    else:
        ic_corr = None

    rank_corr_mean = round(float(np.mean(rank_corrs)), 4) if rank_corrs else None
    rank_corr_std = round(float(np.std(rank_corrs)), 4) if rank_corrs else None

    result = {
        "n_days": len(test_dates) - 1,
        "n_valid_days_t30": len(ic30_valid),
        "n_valid_days_t60": len(ic60_valid),
        "n_common_days": len(rank_corrs),
        "t30": {
            "mean_ic": round(float(np.mean(ic30_valid)), 4) if ic30_valid else None,
            "ic_std": round(float(np.std(ic30_valid)), 4) if ic30_valid else None,
            "ic_pos_pct": round(float(np.mean([v > 0 for v in ic30_valid])), 3) if ic30_valid else None,
        },
        "t60": {
            "mean_ic": round(float(np.mean(ic60_valid)), 4) if ic60_valid else None,
            "ic_std": round(float(np.std(ic60_valid)), 4) if ic60_valid else None,
            "ic_pos_pct": round(float(np.mean([v > 0 for v in ic60_valid])), 3) if ic60_valid else None,
        },
        "ic_correlation": round(ic_corr, 4) if ic_corr is not None else None,
        "score_rank_correlation_mean": rank_corr_mean,
        "score_rank_correlation_std": rank_corr_std,
    }

    # Decision
    if rank_corr_mean is not None:
        if rank_corr_mean > 0.9:
            result["recommendation"] = "SKIP"
            result["reason"] = (
                f"Score rank correlation {rank_corr_mean:.3f} > 0.9 → T=30 and T=60 "
                "produce near-identical rankings. Multi-window ensemble adds no diversity. "
                "Continue with T=60 + Strategy B only."
            )
        elif rank_corr_mean < 0.7:
            result["recommendation"] = "ENSEMBLE"
            result["reason"] = (
                f"Score rank correlation {rank_corr_mean:.3f} < 0.7 → T=30 and T=60 "
                "capture genuinely different signals. Use equal-weight score averaging: "
                "final_score = (score_t30 + score_t60) / 2."
            )
        else:
            result["recommendation"] = "BORDERLINE"
            result["reason"] = (
                f"Score rank correlation {rank_corr_mean:.3f} in (0.7, 0.9). "
                "Moderate diversity. Ensemble may help slightly but gains will be marginal. "
                "Consider 0.3*T30 + 0.7*T60 weighting toward the stronger model."
            )
    else:
        result["recommendation"] = "INCONCLUSIVE"
        result["reason"] = "Insufficient overlapping data days for comparison."

    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    with open(OUT_PATH, "w") as f:
        json.dump(result, f, indent=2)

    print(f"\n{'=' * 60}")
    print(" EVALUATION RESULT")
    print(f"   T=30 mean_IC: {result['t30']['mean_ic']:+.4f}  "
          f"pos_pct: {result['t30']['ic_pos_pct']:.0%}  "
          f"n={result['n_valid_days_t30']}")
    print(f"   T=60 mean_IC: {result['t60']['mean_ic']:+.4f}  "
          f"pos_pct: {result['t60']['ic_pos_pct']:.0%}  "
          f"n={result['n_valid_days_t60']}")
    print(f"   IC correlation:    {result['ic_correlation']:.3f}")
    print(f"   Score rank corr:   {rank_corr_mean:.3f} ± {rank_corr_std:.3f}"
          if rank_corr_mean is not None else f"   Score rank corr: N/A")
    print(f"   Recommendation: {result['recommendation']}")
    print(f"   Reason: {result['reason']}")
    print(f"{'=' * 60}")
    print(f"[eval_t30] Full results -> {OUT_PATH}")


if __name__ == "__main__":
    main()
