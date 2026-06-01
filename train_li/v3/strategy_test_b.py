"""
Strategy B: MOMENTUM STOP-LOSS
===============================
Hypothesis: When CSI300 has dropped significantly over the past few days,
the trend is likely to continue or at least the environment is unfavorable.
Reduce position to 80% on those days.

Tests multiple lookback windows (5d, 10d, 20d) and threshold levels.
Saves results to v3/results/strategy_b.json
"""
import os, json, numpy as np, pandas as pd

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(SCRIPT_DIR, "results", "benchmark_data.parquet")
OUT_PATH  = os.path.join(SCRIPT_DIR, "results", "strategy_b.json")

MIN_POS = 0.80

def sim(rec, weight):
    daily = rec["top20_mean"] * weight
    cum = np.prod(1 + daily / 100)
    mu = daily.mean()
    std = daily.std()
    sharpe = mu / std * np.sqrt(250) if std > 0 else 0
    peak = 1.0; mdd = 0.0
    for d in daily:
        peak = max(peak, peak * (1 + d / 100))
        mdd = min(mdd, (peak * (1 + d / 100)) / peak - 1)
    return float(cum - 1), float(mu), float(std), float(sharpe), float(mdd)

def describe(rec, weight, label):
    daily = rec["top20_mean"] * weight
    cum = 1.0; peak = 1.0; mdd = 0.0
    for d in daily:
        cum *= (1 + d / 100)
        peak = max(peak, cum)
        mdd = min(mdd, cum / peak - 1)
    idx = rec["idx_ret"]
    wins = (daily > idx).sum()
    cum_idx = np.prod(1 + idx / 100)
    return {
        "strategy": label,
        "n_days": len(daily),
        "cum_return": float(cum - 1),
        "cum_return_csi300": float(cum_idx - 1),
        "mean_daily": float(daily.mean()),
        "std_daily": float(daily.std()),
        "sharpe": float(daily.mean() / daily.std() * np.sqrt(250)) if daily.std() > 0 else 0,
        "max_drawdown": float(mdd),
        "excess_vs_csi300": float(cum - cum_idx),
        "beat_csi300_pct": float(wins / len(daily) * 100),
        "worst_day": float(daily.min()),
        "best_day": float(daily.max()),
        "days_at_reduced_position": int((weight < 1.0).sum()),
    }

def main():
    rec = pd.read_parquet(DATA_PATH)
    n = len(rec)
    print(f"[Strategy B] Loaded {n} days from benchmark_data")

    # Baseline
    base = describe(rec, np.ones(n), "BASELINE: always 100% position")
    print(f"  Baseline: ret={base['cum_return']:+.2%} sharpe={base['sharpe']:+.2f} mdd={base['max_drawdown']:+.2%}")

    csi5d = rec["csi5d"].values
    csi10d = rec["csi10d"].values
    csi20d = rec["csi20d"].values
    top20 = rec["top20_mean"].values
    idx = rec["idx_ret"].values

    # ---- Part 1: CSI5D single-threshold sweep ----
    thresholds_5d = [-0.5, -1.0, -1.5, -2.0, -2.5, -3.0, -4.0, -5.0]
    variants_5d = []
    print("\n  CSI 5-day momentum stop:")
    for thr in thresholds_5d:
        weight = np.where(csi5d < thr, MIN_POS, 1.0)
        triggered = int((csi5d < thr).sum())
        stats = describe(rec, weight, f"CSI5d < {thr:+}% -> 80%")
        stats["threshold_pct"] = thr
        stats["days_triggered"] = triggered
        stats["trigger_dates"] = rec.loc[csi5d < thr, "date"].tolist()
        variants_5d.append(stats)
        print(f"    {thr:+5}%: trig={triggered:>2d}d  ret={stats['cum_return']:+.2%}  sharpe={stats['sharpe']:+.2f}  mdd={stats['max_drawdown']:+.2%}")

    # ---- Part 2: CSI10D single-threshold sweep ----
    thresholds_10d = [-1.0, -2.0, -3.0, -4.0, -5.0, -6.0]
    variants_10d = []
    print("\n  CSI 10-day momentum stop:")
    for thr in thresholds_10d:
        weight = np.where(csi10d < thr, MIN_POS, 1.0)
        triggered = int((csi10d < thr).sum())
        stats = describe(rec, weight, f"CSI10d < {thr:+}% -> 80%")
        stats["threshold_pct"] = thr
        stats["days_triggered"] = triggered
        stats["trigger_dates"] = rec.loc[csi10d < thr, "date"].tolist()
        variants_10d.append(stats)
        print(f"    {thr:+5}%: trig={triggered:>2d}d  ret={stats['cum_return']:+.2%}  sharpe={stats['sharpe']:+.2f}  mdd={stats['max_drawdown']:+.2%}")

    # ---- Part 3: CSI20D threshold sweep ----
    thresholds_20d = [-2.0, -3.0, -4.0, -5.0, -6.0, -7.0, -8.0]
    variants_20d = []
    print("\n  CSI 20-day momentum stop:")
    for thr in thresholds_20d:
        weight = np.where(csi20d < thr, MIN_POS, 1.0)
        triggered = int((csi20d < thr).sum())
        stats = describe(rec, weight, f"CSI20d < {thr:+}% -> 80%")
        stats["threshold_pct"] = thr
        stats["days_triggered"] = triggered
        variants_20d.append(stats)
        print(f"    {thr:+5}%: trig={triggered:>2d}d  ret={stats['cum_return']:+.2%}  sharpe={stats['sharpe']:+.2f}  mdd={stats['max_drawdown']:+.2%}")

    # ---- Part 4: Momentum regime decomposition ----
    regime_analysis = []
    for label, mask in [
        ("ALL", slice(None)),
        ("STRONG UP (5d > +3%)", csi5d > 3),
        ("MILD UP (5d 0~+3%)", (csi5d >= 0) & (csi5d <= 3)),
        ("MILD DOWN (5d -3%~0)", (csi5d >= -3) & (csi5d < 0)),
        ("STRONG DOWN (5d < -3%)", csi5d < -3),
        ("DOWN (5d < -1%)", csi5d < -1),
        ("DOWN (5d < -2%)", csi5d < -2),
        ("DOWN (5d < -5%)", csi5d < -5),
    ]:
        sub = rec[mask]
        if len(sub) == 0: continue
        regime_analysis.append({
            "regime": label,
            "n_days": int(len(sub)),
            "top20_mean": float(sub["top20_mean"].mean()),
            "idx_ret_mean": float(sub["idx_ret"].mean()),
            "excess": float(sub["top20_mean"].mean() - sub["idx_ret"].mean()),
            "ic_mean": float(sub["ic"].mean()),
            "worst_top20": float(sub["top20_mean"].min()),
            "worst_date": str(sub.loc[sub["top20_mean"].idxmin(), "date"]),
        })

    # ---- Part 5: Per-disaster-day analysis: would each strategy have saved us? ----
    disaster = rec.loc[rec["top20_mean"] < -2].copy()
    per_disaster = []
    for _, r in disaster.iterrows():
        entry = {
            "date": r["date"],
            "top20_mean": float(r["top20_mean"]),
            "idx_ret": float(r["idx_ret"]),
            "csi5d": float(r["csi5d"]),
            "csi10d": float(r["csi10d"]),
            "csi20d": float(r["csi20d"]),
            "csi20vol": float(r["csi20vol"]),
            "saved_by_CSI5d_lt_neg2": bool(r["csi5d"] < -2),
            "saved_by_CSI5d_lt_neg1": bool(r["csi5d"] < -1),
            "saved_by_CSI5d_lt_neg3": bool(r["csi5d"] < -3),
            "saved_by_CSI10d_lt_neg3": bool(r["csi10d"] < -3),
        }
        per_disaster.append(entry)

    # ---- Part 6: Auto-correlation test ----
    csi5d_vals = rec["csi5d"].values
    next_day = rec["top20_mean"].values
    corr_5d_vs_next = float(np.corrcoef(csi5d_vals, next_day)[0, 1])
    # Test: if CSI5d < 0, mean of next day
    mask_neg5 = csi5d_vals < 0
    mean_when_neg = float(next_day[mask_neg5].mean()) if mask_neg5.sum() > 0 else 0
    mean_when_pos = float(next_day[~mask_neg5].mean()) if (~mask_neg5).sum() > 0 else 0

    # ---- Part 7: Overfitting diagnosis (rolling split) ----
    # Split into first 50% vs last 50% to see if the pattern holds
    half = len(rec) // 2
    first = rec.iloc[:half]
    second = rec.iloc[half:]
    stability = {}
    for tag, sub in [("first_half", first), ("second_half", second)]:
        w_baseline = np.ones(len(sub))
        b = describe(sub, w_baseline, "baseline")
        w_stop = np.where(sub["csi5d"].values < -1.0, MIN_POS, 1.0)
        s = describe(sub, w_stop, "stop")
        stability[tag] = {
            "baseline_ret": b["cum_return"],
            "stop_ret": s["cum_return"],
            "improvement": s["cum_return"] - b["cum_return"],
            "days_triggered": int((sub["csi5d"].values < -1.0).sum()),
        }

    # ==== Output ====
    output = {
        "strategy": "B: Momentum Stop-Loss",
        "data_source": DATA_PATH,
        "comment": "When CSI300 cumulative return over N days drops below threshold, reduce to 80% position. "
                   "Theory: negative momentum tends to persist in the short term; reducing exposure during drawdowns "
                   "limits the impact of crash days where model alpha is overwhelmed by beta.",
        "baseline": base,
        "variants_csi5d": variants_5d,
        "variants_csi10d": variants_10d,
        "variants_csi20d": variants_20d,
        "regime_decomposition": regime_analysis,
        "per_disaster_day_saved": per_disaster,
        "market_autocorrelation": {
            "corr_csi5d_vs_next_day_top20": corr_5d_vs_next,
            "mean_top20_when_csi5d_neg": mean_when_neg,
            "mean_top20_when_csi5d_pos": mean_when_pos,
            "interpretation": f"CSI5d signal has {'POSITIVE' if corr_5d_vs_next>0 else 'NEGATIVE'} correlation with next-day return. "
                              f"Mean top20 return when momentum is negative={mean_when_neg:+.2f}%, positive={mean_when_pos:+.2f}%.",
        },
        "stability_test_train_test_split": stability,
        "config": {
            "min_position": MIN_POS,
            "lookback_windows": ["5d", "10d", "20d"],
            "thresholds_5d": [float(t) for t in thresholds_5d],
            "thresholds_10d": [float(t) for t in thresholds_10d],
            "thresholds_20d": [float(t) for t in thresholds_20d],
            "n_sample_stocks": 600,
        },
    }

    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    with open(OUT_PATH, "w") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    print(f"\n  Results saved to {OUT_PATH}")

    # Quick stability diagnosis
    print(f"\n  STABILITY TEST (first vs second half):")
    for tag, s in stability.items():
        print(f"    {tag}: baseline={s['baseline_ret']:+.2%} stop={s['stop_ret']:+.2%} improvement={s['improvement']:+.2%}")

if __name__ == "__main__":
    main()
