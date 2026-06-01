"""
Strategy A: HIGH-VOLATILITY RISK-OFF
=====================================
Hypothesis: When CSI300 20-day volatility is high, model alpha degrades
and crash risk is elevated. Reduce position to minimum (80%) on those days.

Tests all csi20vol quantile thresholds. Saves results to v3/results/strategy_a.json
"""
import os, json, numpy as np, pandas as pd

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(SCRIPT_DIR, "results", "benchmark_data.parquet")
OUT_PATH  = os.path.join(SCRIPT_DIR, "results", "strategy_a.json")

MIN_POS = 0.80  # competition minimum position

def sim(rec, weight):
    daily = rec["top20_mean"] * weight
    cum = np.prod(1 + daily / 100)
    mu = daily.mean()
    std = daily.std()
    sharpe = mu / std * np.sqrt(250) if std > 0 else 0
    peak = 1; mdd = 0
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
    n = len(daily)
    return {
        "strategy": label,
        "n_days": n,
        "cum_return": float(cum - 1),
        "mean_daily": float(daily.mean()),
        "std_daily": float(daily.std()),
        "sharpe": float(daily.mean() / daily.std() * np.sqrt(250)) if daily.std() > 0 else 0,
        "max_drawdown": float(mdd),
        "excess_vs_csi300": float(cum - np.prod(1 + idx / 100)),
        "beat_csi300_pct": float(wins / n * 100),
        "worst_day": float(daily.min()),
        "best_day": float(daily.max()),
    }

def main():
    rec = pd.read_parquet(DATA_PATH)
    n = len(rec)
    print(f"[Strategy A] Loaded {n} days from benchmark_data")

    vol = rec["csi20vol"].values
    csi_ret = rec["idx_ret"].values
    top20 = rec["top20_mean"].values

    # Baseline
    base = describe(rec, np.ones(n), "BASELINE: always 100% position")
    print(f"  Baseline: ret={base['cum_return']:+.2%} sharpe={base['sharpe']:+.2f} mdd={base['max_drawdown']:+.2%}")

    # Percentile thresholds to test
    thresholds = [60, 65, 70, 75, 80, 85, 90]

    # ---- Part 1: Risk-off to 80% (competition feasible) ----
    variants_80 = []
    for pct in thresholds:
        thr = float(np.percentile(vol, pct))
        weight = np.where(vol > thr, MIN_POS, 1.0)
        triggered = int((vol > thr).sum())
        stats = describe(rec, weight, f"HI-VOL RISK-OFF P{pct} (thr={thr:.4f})")
        stats["percentile"] = pct
        stats["threshold"] = thr
        stats["days_triggered"] = triggered
        stats["trigger_dates"] = rec.loc[vol > thr, "date"].tolist()
        variants_80.append(stats)
        print(f"  P{pct:>2d} -> 80%: triggered {triggered:>2d}d  ret={stats['cum_return']:+.2%}  sharpe={stats['sharpe']:+.2f}  mdd={stats['max_drawdown']:+.2%}")

    # ---- Part 2: Volatility regime decomposition ----
    regime_analysis = []
    for label, mask in [
        ("ALL", slice(None)),
        ("LOW-VOL (csi20vol < P30)", vol < np.percentile(vol, 30)),
        ("MID-VOL (P30-P70)", (vol >= np.percentile(vol, 30)) & (vol <= np.percentile(vol, 70))),
        ("HI-VOL (csi20vol > P70)", vol > np.percentile(vol, 70)),
        ("HI-VOL (csi20vol > P80)", vol > np.percentile(vol, 80)),
        ("HI-VOL (csi20vol > P90)", vol > np.percentile(vol, 90)),
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

    # ---- Part 3: Disaster day analysis ----
    disaster_dates = rec.loc[rec["top20_mean"] < -2, ["date", "top20_mean", "idx_ret", "csi20vol", "csi5d"]]
    disaster_list = []
    for _, r in disaster_dates.iterrows():
        disaster_list.append({
            "date": r["date"],
            "top20_mean": float(r["top20_mean"]),
            "idx_ret": float(r["idx_ret"]),
            "csi20vol": float(r["csi20vol"]),
            "csi5d": float(r["csi5d"]),
            "vol_pct": float((vol > r["csi20vol"]).mean() * 100),
        })

    # ---- Part 4: Risk-off to 0% (what-if, not competition feasible) ----
    variants_cash = []
    for pct in [70, 75, 80, 85, 90]:
        thr = float(np.percentile(vol, pct))
        weight = np.where(vol > thr, 0.0, 1.0)
        stats = describe(rec, weight, f"HI-VOL FULL CASH P{pct}")
        stats["percentile"] = pct
        variants_cash.append(stats)

    # ---- Part 5: Combined signals (vol + momentum) ----
    combined = []
    for vp in [70, 80]:
        for mp in [-1, -2, -3]:
            vthr = float(np.percentile(vol, vp))
            weight = np.where((vol > vthr) | (rec["csi5d"].values < mp), MIN_POS, 1.0)
            triggered = int(((vol > vthr) | (rec["csi5d"].values < mp)).sum())
            stats = describe(rec, weight, f"VOL>{vp} OR CSI5D<{mp}")
            stats["days_triggered"] = triggered
            combined.append(stats)

    # ==== Assemble output ====
    output = {
        "strategy": "A: High-Volatility Risk-Off",
        "data_source": DATA_PATH,
        "comment": "When CSI300 20d volatility exceeds quantile threshold, reduce to 80% position (competition minimum). Theory: high-vol regimes concentrate tail risk; model alpha doesn't compensate for the extra beta exposure on crash days.",
        "baseline": base,
        "variants_80pct": variants_80,
        "regime_decomposition": regime_analysis,
        "disaster_days_top20_lt_neg2pct": disaster_list,
        "variants_full_cash_for_reference_only": variants_cash,
        "combined_vol_momentum": combined,
        "config": {
            "min_position": MIN_POS,
            "vol_window": "csi20vol (std of daily returns over 20 trading days)",
            "n_thresholds_tested": len(thresholds),
            "n_sample_stocks": 600,
        },
    }

    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    with open(OUT_PATH, "w") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    print(f"\n  Results saved to {OUT_PATH}")

if __name__ == "__main__":
    main()
