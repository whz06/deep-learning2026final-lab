"""v4/feasibility.py — Multi-window ensemble feasibility check

Layer 1: Model-free momentum alpha proxy → IC correlation across lookback windows
  - mom_5d, mom_20d, mom_60d: cross-sectional rank IC vs T+1 return
  - If IC time series are uncorrelated across windows → diversity is real

Layer 2: T=60 GRU model on W=60 vs W=90
  - Daily Rank IC for each window
  - Score rank correlation between windows on common stocks
  - W=30 skipped: T=60 model's linear head is calibrated for 60-step hidden states

Output: v4/results/feasibility.json
"""

import os, sys, json
import numpy as np
import pandas as pd
import torch
from scipy.stats import spearmanr

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(SCRIPT_DIR)
V2_DIR = os.path.join(ROOT, "v2")
PARQUET_PATH = os.path.join(ROOT, "processed", "all_data.parquet")
INDEX_PATH = os.path.join(ROOT, "data", "market", "000300.SH.csv")
CKPT_DIR = os.path.join(V2_DIR, "checkpoints")
OUT_PATH = os.path.join(SCRIPT_DIR, "results", "feasibility.json")

sys.path.insert(0, V2_DIR)
from models.gru import GRURanker

RAW = ["open", "high", "low", "close", "vol", "amount",
       "pct_chg", "turnover_rate", "volume_ratio", "total_mv"]
TECH = ["macd", "macd_signal", "rsi", "bb_width", "bb_pct", "mom_5", "mom_20", "vol_20"]
CKPT = "gru_gru_hidden_size=128_num_layers=1_dropout=0.2_lr=0.0003.pt"


def add_tech(df):
    """Add 8 technical indicators to stock dataframe (on full history)."""
    c = df["close"].astype(float)
    e12 = c.ewm(span=12, adjust=False).mean()
    e26 = c.ewm(span=26, adjust=False).mean()
    df["macd"] = e12 - e26
    df["macd_signal"] = df["macd"].ewm(span=9, adjust=False).mean()

    d = c.diff()
    g = d.clip(lower=0)
    l = (-d).clip(lower=0)
    rs = g.ewm(alpha=1/14, adjust=False).mean() / (l.ewm(alpha=1/14, adjust=False).mean() + 1e-8)
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
    """Build normalized features of lookback=W for all stocks on a given date."""
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


def compute_mom_ic(date, next_date, series, W):
    """Spearman rank IC of W-day close momentum vs next-day pct_chg.
    Uses cross-sectional rank of raw momentum (not percentile-normalized)."""
    moms, rets = [], []
    for ts, sdf in series.items():
        sd = sdf[sdf["trade_date"] <= date]
        if len(sd) < W + 1:
            continue
        close_w = sd["close"].values[-W - 1]
        close_t = sd["close"].values[-1]
        if close_w <= 0:
            continue
        mom = close_t / close_w - 1

        nr = sdf[sdf["trade_date"] == next_date]
        if len(nr) == 0:
            continue
        ret = nr["pct_chg"].values[0]
        moms.append(mom)
        rets.append(ret)

    if len(moms) < 30:
        return None, 0
    ic = spearmanr(moms, rets).correlation
    return ic, len(moms)


def compute_gru_ic(model, tensor, codes, next_date, series, device):
    """Run GRU on features, compute per-stock next-day return, return IC."""
    scores = model(tensor.to(device)).detach().cpu().numpy()
    sl, lb = [], []
    for ci, cc in enumerate(codes):
        r = series[cc][series[cc]["trade_date"] == next_date]
        if len(r) > 0:
            sl.append(scores[ci])
            lb.append(r["pct_chg"].values[0])
    if len(sl) < 30:
        return None, 0
    ic = spearmanr(sl, lb).correlation
    return ic, len(sl)


def pairwise_corr(a, b):
    """Pearson correlation of two time series, dropping None pairs."""
    mask = [i for i in range(len(a)) if a[i] is not None and b[i] is not None]
    if len(mask) < 5:
        return None
    aa = [a[i] for i in mask]
    bb = [b[i] for i in mask]
    return float(np.corrcoef(aa, bb)[0, 1])


def summarize(ics):
    """Mean, std, positive-pct from list of (ic, n_stocks) tuples."""
    vals = [v for v, _ in ics if v is not None]
    if not vals:
        return {"mean_ic": None, "ic_std": None, "ic_pos_pct": None, "n_valid_days": 0}
    return {
        "mean_ic": round(float(np.mean(vals)), 4),
        "ic_std": round(float(np.std(vals)), 4),
        "ic_pos_pct": round(float(np.mean([v > 0 for v in vals])), 3),
        "n_valid_days": len(vals),
    }


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[feasibility] Device: {device}")

    model = GRURanker(22, 128, 1, 0.2)
    model.load_state_dict(torch.load(os.path.join(CKPT_DIR, CKPT),
                         map_location=device, weights_only=True))
    model.to(device).eval()

    df = pd.read_parquet(PARQUET_PATH)
    df["trade_date"] = df["trade_date"].astype(str)
    dates_all = sorted(df["trade_date"].unique())
    test_start_idx = [i for i, d in enumerate(dates_all) if d >= "20260201"][0]
    lb = max(0, test_start_idx - 90 - 1 - 20)
    df = df[df["trade_date"] >= dates_all[lb]]
    print(f"[feasibility] Data rows: {len(df):,}")

    csi = pd.read_csv(INDEX_PATH, dtype={"trade_date": str})
    idx_map = dict(zip(csi["trade_date"], csi["pct_chg"]))

    stocks = sorted(df["ts_code"].unique())
    np.random.seed(42)
    stocks = sorted(np.random.choice(stocks, 600, replace=False))
    print(f"[feasibility] {len(stocks)} stocks sampled")

    series = {ts: add_tech(df[df["ts_code"] == ts].sort_values("trade_date")
                .reset_index(drop=True)) for ts in stocks}

    test_dates = [d for d in dates_all if "20260201" <= d <= "20260531"]
    print(f"[feasibility] Test dates: {len(test_dates)} ({test_dates[0]} -> {test_dates[-1]})")

    # --- Collect daily ICs ---
    ic_mom_5d, ic_mom_20d, ic_mom_60d = [], [], []
    ic_gru_60, ic_gru_90 = [], []
    score_rank_corrs = []

    for di, date in enumerate(test_dates):
        if di == len(test_dates) - 1:
            break
        nd = test_dates[di + 1]

        # Layer 1: Momentum ICs
        mi5, n5 = compute_mom_ic(date, nd, series, 5)
        ic_mom_5d.append((mi5, n5))
        mi20, n20 = compute_mom_ic(date, nd, series, 20)
        ic_mom_20d.append((mi20, n20))
        mi60, n60 = compute_mom_ic(date, nd, series, 60)
        ic_mom_60d.append((mi60, n60))

        # Layer 2: GRU W=60
        f60, c60 = build(date, series, idx_map, 60)
        if f60 is not None:
            g60, ng60 = compute_gru_ic(model, torch.from_numpy(f60), c60, nd, series, device)
            ic_gru_60.append((g60, ng60))
        else:
            ic_gru_60.append((None, 0))

        # Layer 2: GRU W=90
        f90, c90 = build(date, series, idx_map, 90)
        if f90 is not None:
            g90, ng90 = compute_gru_ic(model, torch.from_numpy(f90), c90, nd, series, device)
            ic_gru_90.append((g90, ng90))

            # Score rank correlation between W=60 and W=90 on common stocks
            if f60 is not None:
                common = list(set(c60) & set(c90))
                if len(common) >= 30:
                    idx60 = [c60.index(c) for c in common]
                    idx90 = [c90.index(c) for c in common]
                    s60 = model(torch.from_numpy(f60).to(device)).detach().cpu().numpy()
                    s90 = model(torch.from_numpy(f90).to(device)).detach().cpu().numpy()
                    sr = spearmanr(s60[idx60], s90[idx90]).correlation
                    score_rank_corrs.append(sr)
        else:
            ic_gru_90.append((None, 0))

        torch.cuda.empty_cache()

        if (di + 1) % 20 == 0:
            print(f"  ... processed {di + 1}/{len(test_dates) - 1} days", flush=True)

    # --- Aggregate ---
    l1_ics = {
        "mom_5d":  [v for v, _ in ic_mom_5d],
        "mom_20d": [v for v, _ in ic_mom_20d],
        "mom_60d": [v for v, _ in ic_mom_60d],
    }

    result = {
        "n_days": len(test_dates) - 1,
        "date_range": [test_dates[0], test_dates[-1]],
        "layer1": {
            "mom_5d":  summarize(ic_mom_5d),
            "mom_20d": summarize(ic_mom_20d),
            "mom_60d": summarize(ic_mom_60d),
            "ic_correlations": {
                "5d_vs_20d": pairwise_corr(l1_ics["mom_5d"], l1_ics["mom_20d"]),
                "5d_vs_60d": pairwise_corr(l1_ics["mom_5d"], l1_ics["mom_60d"]),
                "20d_vs_60d": pairwise_corr(l1_ics["mom_20d"], l1_ics["mom_60d"]),
            },
        },
        "layer2": {
            "gru_w60": summarize(ic_gru_60),
            "gru_w90": summarize(ic_gru_90),
            "ic_correlation": pairwise_corr(
                [v for v, _ in ic_gru_60], [v for v, _ in ic_gru_90]
            ),
            "score_rank_correlation_mean": (
                round(float(np.mean(score_rank_corrs)), 4)
                if score_rank_corrs else None
            ),
            "score_rank_correlation_std": (
                round(float(np.std(score_rank_corrs)), 4)
                if score_rank_corrs else None
            ),
            "n_common_days": len(score_rank_corrs),
        },
    }

    # --- Recommendation ---
    avg_corr_l1 = np.mean([v for v in result["layer1"]["ic_correlations"].values()
                           if v is not None])
    avg_rank_corr_l2 = result["layer2"]["score_rank_correlation_mean"]

    if avg_corr_l1 is not None and avg_corr_l1 < 0.3 and avg_rank_corr_l2 is not None and avg_rank_corr_l2 < 0.5:
        result["recommendation"] = "TRAIN"
        result["reason"] = (
            f"Low IC correlation across windows (mean {avg_corr_l1:.2f}) "
            f"and low score rank correlation ({avg_rank_corr_l2:.2f}) "
            f"→ different lookback windows capture complementary alpha. "
            f"Worth training T=30 and T=90 GRU models."
        )
    elif avg_corr_l1 is not None and avg_corr_l1 > 0.5:
        result["recommendation"] = "SKIP"
        result["reason"] = (
            f"High IC correlation across windows (mean {avg_corr_l1:.2f}) "
            f"→ different lookback windows see the same signal. "
            f"Multi-window ensemble unlikely to add meaningful diversity."
        )
    else:
        result["recommendation"] = "BORDERLINE"
        result["reason"] = (
            f"Mixed signals: L1 IC corr mean={avg_corr_l1}, "
            f"L2 rank corr mean={avg_rank_corr_l2}. "
            f"Proceed with training but keep T=60 as fallback."
        )

    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    with open(OUT_PATH, "w") as f:
        json.dump(result, f, indent=2, default=str)

    print(f"\n[feasibility] Results saved to {OUT_PATH}")
    print(f"[feasibility] Recommendation: {result['recommendation']}")
    print(f"[feasibility] Reason: {result['reason']}")

    # Print summary
    print("\n" + "=" * 60)
    print(" LAYER 1: Momentum Alpha Proxy IC")
    for k in ["mom_5d", "mom_20d", "mom_60d"]:
        s = result["layer1"][k]
        print(f"   {k:>8}: mean_IC={s['mean_ic']:+.4f}  std={s['ic_std']:.4f}  "
              f"pos_pct={s['ic_pos_pct']:.0%}  n={s['n_valid_days']}")
    print(f"   IC correlations: 5d-20d={result['layer1']['ic_correlations']['5d_vs_20d']:.3f}"
          f"  5d-60d={result['layer1']['ic_correlations']['5d_vs_60d']:.3f}"
          f"  20d-60d={result['layer1']['ic_correlations']['20d_vs_60d']:.3f}")
    print(f"\n LAYER 2: GRU Model IC")
    for k in ["gru_w60", "gru_w90"]:
        s = result["layer2"][k]
        print(f"   {k:>8}: mean_IC={s['mean_ic']:+.4f}  std={s['ic_std']:.4f}  "
              f"pos_pct={s['ic_pos_pct']:.0%}  n={s['n_valid_days']}")
    print(f"   IC correlation W60-W90: {result['layer2']['ic_correlation']:.3f}")
    print(f"   Score rank corr (common stocks): {avg_rank_corr_l2:.3f} ± "
          f"{result['layer2']['score_rank_correlation_std']:.3f}"
          f"  (n={result['layer2']['n_common_days']})")
    print("=" * 60)


if __name__ == "__main__":
    main()
