"""Exploratory analysis: score dispersion vs crash days.
Answers: Is model "confidence" (score_std) predictive of crash risk?
"""
import os, json, numpy as np, pandas as pd

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(SCRIPT_DIR, "results", "benchmark_data.parquet")
OUT_PATH  = os.path.join(SCRIPT_DIR, "results", "dispersion_analysis.json")

def main():
    rec = pd.read_parquet(DATA_PATH)
    n = len(rec)
    print(f"[Dispersion Analysis] {n} days loaded")

    sc_std = rec["score_std"].values
    sc_skew = rec["score_skew"].values
    sc_range = rec["score_min"].values
    sc_max = rec["score_max"].values
    t20_mean = rec["top20_mean"].values
    t20_std = rec["top20_std"].values
    n_z2 = rec["n_above_z2"].values
    top20_sc_range = rec["top20_score_range"].values
    top20_sc_std = rec["top20_score_std"].values
    dates = rec["date"].values

    # ======== 1. Basic correlations ========
    print("\n  Correlations with next-day top20 return:")
    for name, vals in [
        ("score_std", sc_std),
        ("score_skew", sc_skew),
        ("n_above_z2", n_z2),
        ("top20_score_range", top20_sc_range),
        ("top20_score_std", top20_sc_std),
        ("score_range(all)", sc_max - rec["score_min"].values),
    ]:
        c = float(np.corrcoef(vals, t20_mean)[0, 1])
        print(f"    {name:>25}: r={c:+.4f}")

    # ======== 2. Score_std quantile analysis ========
    print(f"\n  Score_std quantiles:")
    for q in [10, 25, 50, 75, 85, 90, 95]:
        print(f"    P{q:>2}: {np.percentile(sc_std, q):.4f}")

    # ======== 3. Crash vs non-crash score distribution ========
    crash_mask = t20_mean < -2.0
    normal_mask = (t20_mean >= -2.0) & (t20_mean <= 2.0)
    bull_mask = t20_mean > 2.0

    print(f"\n  Score_std by market outcome:")
    for label, mask in [("CRASH (<-2%)", crash_mask), ("NEUTRAL", normal_mask), ("BULL (>+2%)", bull_mask)]:
        if mask.sum() == 0: continue
        ss = sc_std[mask]
        print(f"    {label:>15} (n={mask.sum():>2}): mean={ss.mean():.4f}  median={np.median(ss):.4f}  "
              f"P10={np.percentile(ss,10):.4f}  P90={np.percentile(ss,90):.4f}")

    # ======== 4. HIGH score_std days → what happened? ========
    print(f"\n  Days with score_std > P90 ({np.percentile(sc_std, 90):.4f}):")
    hi_std = sc_std >= np.percentile(sc_std, 90)
    sub = rec[hi_std].sort_values("top20_mean")
    for _, r in sub.iterrows():
        is_crash = "★CRASH" if r["top20_mean"] < -2 else ""
        print(f"    {r['date']}  top20={r['top20_mean']:+6.2f}%  sc_std={r['score_std']:.4f}  "
              f"sc_skew={r['score_skew']:+.3f}  n_z2={r['n_above_z2']}  csi5d={r['csi5d']:+.1f}%  "
              f"{is_crash}")

    # ======== 5. LOW score_std days → what happened? ========
    print(f"\n  Days with score_std < P10 ({np.percentile(sc_std, 10):.4f}):")
    lo_std = sc_std <= np.percentile(sc_std, 10)
    sub = rec[lo_std].sort_values("top20_mean")
    for _, r in sub.iterrows():
        print(f"    {r['date']}  top20={r['top20_mean']:+6.2f}%  sc_std={r['score_std']:.4f}  "
              f"sc_skew={r['score_skew']:+.3f}")

    # ======== 6. Key: what about the two crash days strategy B missed? ========
    print(f"\n  === CRITICAL: The 2 crash days strategy B missed ===")
    for target_date in ["20260529", "20260521"]:
        row = rec[rec["date"] == target_date]
        if len(row) == 0: continue
        r = row.iloc[0]
        pct_std = float((sc_std > r["score_std"]).mean() * 100)
        pct_skew = float((sc_skew > r["score_skew"]).mean() * 100)
        pct_range = float((top20_sc_range > r["top20_score_range"]).mean() * 100)
        print(f"    {target_date}: top20={r['top20_mean']:+.2f}%  csi5d={r['csi5d']:+.1f}%")
        print(f"      score_std={r['score_std']:.4f}  (>{pct_std:.0f}% of days)")
        print(f"      score_skew={r['score_skew']:+.4f}  (>{pct_skew:.0f}% of days)")
        print(f"      top20_score_range={r['top20_score_range']:.4f}  (>{pct_range:.0f}% of days)")
        print(f"      n_above_z2={r['n_above_z2']}  ic={r['ic']:+.4f}")
        print(f"      → Would score_std > P90 trigger? {'YES' if pct_std >= 90 else 'NO'}")
        print(f"      → Would score_skew > P90 trigger? {'YES' if pct_skew >= 90 else 'NO'}")

    # ======== 7. Score_skew distribution: overconfidence signal ========
    print(f"\n  Score_skew quantiles:")
    for q in [10, 25, 50, 75, 85, 90, 95]:
        print(f"    P{q:>2}: {np.percentile(sc_skew, q):+.4f}")

    hi_skew_mask = sc_skew >= np.percentile(sc_skew, 90)
    hi_skew_sub = rec[hi_skew_mask]
    print(f"  High-skew (>P90, n={hi_skew_mask.sum()}) top20 mean return: {hi_skew_sub['top20_mean'].mean():+.2f}%")
    print(f"  Low-skew  (<P10, n={(sc_skew <= np.percentile(sc_skew, 10)).sum()}) top20 mean return: "
          f"{rec[sc_skew <= np.percentile(sc_skew, 10)]['top20_mean'].mean():+.2f}%")

    # ======== 8. What's the single best dispersion metric for defense? ========
    print(f"\n  === BEST DISPERSION METRIC FOR DEFENSE ===")
    best_metric, best_diff = None, 0
    for name, vals in [
        ("score_std", sc_std),
        ("score_skew", sc_skew),
        ("top20_score_range", top20_sc_range),
        ("top20_score_std", top20_sc_std),
        ("n_above_z2", n_z2.astype(float)),
    ]:
        # If we exclude top 20% of this metric, what's the mean?
        hi = vals >= np.percentile(vals, 80)
        lo = vals <= np.percentile(vals, 20)
        nd_hi = len(hi)
        diff = float(rec[lo]["top20_mean"].mean() - rec[hi]["top20_mean"].mean())
        print(f"    {name:>20}: Top20={rec[hi]['top20_mean'].mean():+.2f}%  Bot20={rec[lo]['top20_mean'].mean():+.2f}%  diff={diff:+.2f}%")

    # Save summary
    output = {
        "crash_days": rec.loc[crash_mask, ["date","top20_mean","score_std","score_skew","csi5d"]].to_dict("records"),
        "correlations": {name: float(np.corrcoef(vals, t20_mean)[0,1])
                         for name, vals in [("score_std",sc_std),("score_skew",sc_skew),
                                            ("top20_score_std",top20_sc_std),("n_above_z2",n_z2.astype(float))]},
    }
    with open(OUT_PATH, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\n  Analysis saved to {OUT_PATH}")

if __name__ == "__main__":
    main()
