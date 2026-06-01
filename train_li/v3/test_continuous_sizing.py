"""
Test continuous position sizing (Direction 4).
Compare linear, exponential, and sigmoid mappings against binary threshold.
Saves to v3/results/strategy_continuous.json
"""
import os, json, numpy as np, pandas as pd

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(SCRIPT_DIR, "results", "benchmark_data.parquet")
OUT_PATH  = os.path.join(SCRIPT_DIR, "results", "strategy_continuous.json")

MIN_POS = 0.80
MAX_POS = 1.00

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
        "worst_day": float(daily.min()),
        "best_day": float(daily.max()),
        "mean_weight": float(weight.mean()),
        "min_weight": float(weight.min()),
        "days_below_full": int((weight < 1.0).sum()),
    }

def main():
    rec = pd.read_parquet(DATA_PATH)
    n = len(rec)
    csi5d = rec["csi5d"].values
    csi10d = rec["csi10d"].values
    csi20vol = rec["csi20vol"].values
    print(f"[Continuous Sizing] {n} days")

    baseline = simulate(rec, np.ones(n), "BASELINE: 100%")
    w_bin = np.where(csi5d < -1.0, MIN_POS, MAX_POS)
    strat_b = simulate(rec, w_bin, "BINARY: csi5d<-1% -> 80%")
    print(f"  Baseline:    ret={baseline['cum_return']:+.2%}  sharpe={baseline['sharpe']:+.2f}  mdd={baseline['max_drawdown']:+.2%}")
    print(f"  Binary(strB): ret={strat_b['cum_return']:+.2%}  sharpe={strat_b['sharpe']:+.2f}  mdd={strat_b['max_drawdown']:+.2%}  mean_w={strat_b['mean_weight']:.3f}")

    # ---- 1. Linear mapping: pos = clamp(0.8, 1 + csi5d/k, 1) ----
    linear = []
    print("\n  LINEAR pos = clamp(0.8, 1 + csi5d/k, 1):")
    for k in [15, 20, 25, 30, 35, 40, 50]:
        weight = np.clip(1.0 + csi5d / k, MIN_POS, MAX_POS)
        s = simulate(rec, weight, f"LINEAR k={k}")
        s["k"] = k
        linear.append(s)
        print(f"    k={k:>2}: ret={s['cum_return']:+.2%}  sharpe={s['sharpe']:+.2f}  mdd={s['max_drawdown']:+.2%}  mean_w={s['mean_weight']:.3f}  low_d={s['days_below_full']}")

    # ---- 2. Exponential: pos = clamp(0.8, exp(csi5d/lam), 1) ----
    expo = []
    print("\n  EXPONENTIAL pos = clamp(0.8, exp(csi5d/lam), 1):")
    for lam in [20, 25, 30, 35, 40]:
        weight = np.clip(np.exp(csi5d / lam), MIN_POS, MAX_POS)
        s = simulate(rec, weight, f"EXP lambda={lam}")
        s["lambda"] = lam
        expo.append(s)
        print(f"    lam={lam:>2}: ret={s['cum_return']:+.2%}  sharpe={s['sharpe']:+.2f}  mdd={s['max_drawdown']:+.2%}  mean_w={s['mean_weight']:.3f}  low_d={s['days_below_full']}")

    # ---- 3. Piecewise linear ----
    pw_linear = []
    print("\n  PIECEWISE: full above upper, linear below lower, 80% at bottom:")
    for upper in [-0.5, 0.0, +0.5]:
        for lower in [-3.0, -4.0, -5.0, -6.0]:
            weight = np.ones(n)
            for i in range(n):
                if csi5d[i] >= upper:
                    weight[i] = 1.0
                elif csi5d[i] <= lower:
                    weight[i] = 0.8
                else:
                    frac = (csi5d[i] - lower) / (upper - lower)
                    weight[i] = 0.8 + 0.2 * frac
            s = simulate(rec, weight, f"PW upper={upper:+} lower={lower:+}")
            s["upper"] = upper; s["lower"] = lower
            pw_linear.append(s)
            if abs(s["cum_return"] - strat_b["cum_return"]) > 0.002 or s["sharpe"] > strat_b["sharpe"]:
                print(f"    u={upper:+} l={lower:+}: ret={s['cum_return']:+.2%}  sharpe={s['sharpe']:+.2f}  mdd={s['max_drawdown']:+.2%}  mean_w={s['mean_weight']:.3f}  low_d={s['days_below_full']}")

    # ---- 4. Vol-adaptive: base weight * vol_scaling ----
    vol_adaptive = []
    print("\n  VOL-ADAPTIVE: base(csi5d) * vol_ref/csi20vol:")
    # Base weight from linear k=25
    base_w = np.clip(1.0 + csi5d / 25, MIN_POS, MAX_POS)
    # Vol reference levels
    vol_refs = [0.008, 0.010, 0.012, 0.015, 0.020]
    for vref in vol_refs:
        vol_scale = np.clip(vref / (csi20vol + 1e-8), 0.7, 1.2)
        weight = np.clip(base_w * vol_scale, MIN_POS, 1.05)
        weight = np.minimum(weight, MAX_POS)
        s = simulate(rec, weight, f"VOL-ADAPTIVE vref={vref:.4f}")
        s["vol_ref"] = vref
        vol_adaptive.append(s)
        print(f"    vref={vref:.4f}: ret={s['cum_return']:+.2%}  sharpe={s['sharpe']:+.2f}  mdd={s['max_drawdown']:+.2%}  mean_w={s['mean_weight']:.3f}  low_d={s['days_below_full']}")

    # ---- 5. Comparison table ----
    all_strategies = [baseline, strat_b] + linear + expo + vol_adaptive
    best = max(all_strategies, key=lambda x: (x["sharpe"] + (x["cum_return"]>0)*0.5))
    best_cum = max(all_strategies, key=lambda x: x["cum_return"])

    print(f"\n  {'='*75}")
    print(f"  BEST SHARPE: {best['strategy']}: ret={best['cum_return']:+.2%} sharpe={best['sharpe']:+.2f}")
    print(f"  BEST RETURN: {best_cum['strategy']}: ret={best_cum['cum_return']:+.2%} sharpe={best_cum['sharpe']:+.2f}")

    # ---- 6. Combined: best continuous + dispersion ----
    # Just for the output file
    print(f"\n  Binary  days_below_full={strat_b['days_below_full']}")
    print(f"  Linear  k=25: days_below_full={[s for s in linear if s['k']==25][0]['days_below_full']}")

    output = {
        "strategy": "Continuous Position Sizing",
        "baseline": baseline,
        "binary_strategy_b": strat_b,
        "linear_variants": linear,
        "exponential_variants": expo,
        "piecewise_linear_variants": pw_linear,
        "vol_adaptive_variants": vol_adaptive,
        "best_sharpe": best,
        "best_return": best_cum,
        "improvement_over_binary": best["cum_return"] - strat_b["cum_return"],
    }
    with open(OUT_PATH, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\n  Saved to {OUT_PATH}")

if __name__ == "__main__":
    main()
