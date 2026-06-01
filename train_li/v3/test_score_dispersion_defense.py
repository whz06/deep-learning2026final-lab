"""
Test score-dispersion defense triggers. Compare against baseline and
strategy B (momentum stop). Saves to v3/results/strategy_dispersion.json
"""
import os, json, numpy as np, pandas as pd

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(SCRIPT_DIR, "results", "benchmark_data.parquet")
OUT_PATH  = os.path.join(SCRIPT_DIR, "results", "strategy_dispersion.json")

MIN_POS = 0.80

def simulate(rec, weight, label):
    daily = rec["top20_mean"] * weight
    cum = 1.0; peak = 1.0; mdd = 0.0
    for d in daily: cum*=(1+d/100); peak=max(peak,cum); mdd=min(mdd, cum/peak-1)
    idx_cum = np.prod(1 + rec["idx_ret"]/100)
    wins = (daily > rec["idx_ret"]).sum()
    n = len(daily)
    return {
        "strategy": label, "n_days": n,
        "cum_return": float(cum-1),
        "cum_csi300": float(idx_cum-1),
        "mean_daily": float(daily.mean()),
        "std_daily": float(daily.std()),
        "sharpe": float(daily.mean()/daily.std()*np.sqrt(250)) if daily.std()>0 else 0,
        "max_drawdown": float(mdd),
        "excess_vs_csi300": float(cum-idx_cum),
        "beat_csi300_pct": float(wins/n*100),
        "worst_day": float(daily.min()),
        "days_at_reduced": int((weight<1.0).sum()),
        "trigger_dates": sorted(set(rec.loc[weight<1.0, "date"].tolist())),
    }

def main():
    rec = pd.read_parquet(DATA_PATH)
    n = len(rec)
    print(f"[Dispersion Defense] {n} days")

    sc_std = rec["score_std"].values
    sc_skew = rec["score_skew"].values
    n_z2 = rec["n_above_z2"].values
    top20_sc_std = rec["top20_score_std"].values
    top20_sc_range = rec["top20_score_range"].values
    csi5d = rec["csi5d"].values
    dates = rec["date"].values

    baseline = simulate(rec, np.ones(n), "BASELINE: always 100%")
    # Best strategy B from earlier
    w_b = np.where(csi5d < -1.0, MIN_POS, 1.0)
    strat_b = simulate(rec, w_b, "STRATEGY_B: csi5d<-1% -> 80%")
    print(f"  Baseline:   ret={baseline['cum_return']:+.2%}  sharpe={baseline['sharpe']:+.2f}  mdd={baseline['max_drawdown']:+.2%}")
    print(f"  Strategy B: ret={strat_b['cum_return']:+.2%}  sharpe={strat_b['sharpe']:+.2f}  mdd={strat_b['max_drawdown']:+.2%}")

    # ---- Part 1: Pure score_std single trigger ----
    variants = []
    for pct_tag, pct in [("P80",80), ("P85",85), ("P90",90), ("P92",92), ("P95",95), ("P97",97), ("P99",99)]:
        thr = float(np.percentile(sc_std, pct))
        weight = np.where(sc_std > thr, MIN_POS, 1.0)
        stats = simulate(rec, weight, f"sc_std>{pct_tag}({thr:.4f})")
        stats["percentile"] = pct; stats["threshold"] = thr
        trigger_crash = set(stats["trigger_dates"]) & set(rec.loc[rec["top20_mean"]<-2, "date"].tolist())
        stats["crash_days_saved"] = len(trigger_crash)
        stats["crash_dates_saved"] = sorted(trigger_crash)
        variants.append(stats)

    # ---- Part 2: score_skew triggers ----
    for pct_tag, pct in [("P5",5), ("P10",10), ("P15",15)]:
        thr = float(np.percentile(sc_skew, pct))
        weight = np.where(sc_skew < thr, MIN_POS, 1.0)
        stats = simulate(rec, weight, f"sc_skew<{pct_tag}({thr:+.4f})")
        stats["percentile"] = pct; stats["threshold"] = thr

    # ---- Part 3: Combined trigger: score_std high OR score_std low ----
    for hi_pct, lo_pct in [(95,5), (90,10), (95,10)]:
        hi_t = float(np.percentile(sc_std, hi_pct))
        lo_t = float(np.percentile(sc_std, lo_pct))
        weight = np.where((sc_std > hi_t) | (sc_std < lo_t), MIN_POS, 1.0)
        stats = simulate(rec, weight, f"sc_std>(P{hi_pct}|<P{lo_pct})")
        stats["high_threshold"] = hi_t; stats["low_threshold"] = lo_t

    # ---- Part 4: Combined with momentum ----
    combos = []
    for hi_pct in [90, 92, 95, 97]:
        hi_t = float(np.percentile(sc_std, hi_pct))
        for lo_pct in [5, 10]:
            lo_t = float(np.percentile(sc_std, lo_pct))
            weight = np.where((csi5d < -1.0) | (sc_std > hi_t) | (sc_std < lo_t), MIN_POS, 1.0)
            stats = simulate(rec, weight,
                f"MOMENTUM(csi5d<-1%) + DISPERSION(sc_std>P{hi_pct}|<P{lo_pct})")
            stats["hi_pct"] = hi_pct; stats["lo_pct"] = lo_pct
            combos.append(stats)

    # ---- Part 5: The best combo ----
    print(f"\n  {'Strategy':<55} {'Trigger':>3}d  {'Ret':>7}  {'Sharpe':>6}  {'MDD':>7}  {'CrashSaved':>10}")
    print("-"*95)
    for s in [baseline, strat_b] + variants[:5]:
        cs = s.get("crash_days_saved", 0)
        print(f"  {s['strategy']:<55} {s['days_at_reduced']:>3}  {s['cum_return']:>+6.2%}  {s['sharpe']:>+5.2f}  {s['max_drawdown']:>+6.2%}  {cs:>10}")

    print("\n  COMBINED (momentum + dispersion):")
    best_combo = max(combos, key=lambda x: x["cum_return"])
    for s in sorted(combos, key=lambda x: x["cum_return"], reverse=True):
        cs = len(set(s["trigger_dates"]) & set(rec.loc[rec["top20_mean"]<-2, "date"].tolist()))
        marker = " ← BEST" if s is best_combo else ""
        print(f"  {s['strategy']:<55} {s['days_at_reduced']:>3}  {s['cum_return']:>+6.2%}  {s['sharpe']:>+5.2f}  {s['max_drawdown']:>+6.2%}  {cs:>10}{marker}")

    # ---- Part 6: Specific crash day analysis ----
    crash_dates = rec.loc[rec["top20_mean"] < -2, ["date", "top20_mean", "csi5d", "score_std"]]
    print(f"\n{'='*75}")
    print("CRASH DAY DETAIL: which trigger catches each?")
    print(f"{'='*75}")
    print(f"{'Date':<12} {'top20':>7} {'csi5d':>6} {'sc_std':>7} {'StratB?':>8} {'P95?':>6} {'P92?':>6}")
    print("-"*52)
    p95_t = float(np.percentile(sc_std, 95))
    p92_t = float(np.percentile(sc_std, 92))
    for _, r in crash_dates.iterrows():
        b_trig = r["csi5d"] < -1.0
        p95_trig = r["score_std"] > p95_t
        p92_trig = r["score_std"] > p92_t
        print(f"  {r['date']:<12} {r['top20_mean']:>+6.2f}% {r['csi5d']:>+5.1f}% {r['score_std']:>7.4f} "
              f"{'YES' if b_trig else 'no':>8} {'YES' if p95_trig else 'no':>6} {'YES' if p92_trig else 'no':>6}")

    # ---- Save ----
    output = {
        "strategy": "Score Dispersion Defense",
        "baseline": baseline,
        "strategy_b_reference": strat_b,
        "pure_score_std_variants": variants,
        "combined_momentum_dispersion": combos,
        "best_combo": best_combo,
        "improvement_over_baseline": best_combo["cum_return"] - baseline["cum_return"],
        "improvement_over_strat_b": best_combo["cum_return"] - strat_b["cum_return"],
        "crash_days_detail": crash_dates.to_dict("records"),
    }
    with open(OUT_PATH, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\n  Saved to {OUT_PATH}")

if __name__ == "__main__":
    main()
