"""
Deep analysis: WHY is dispersion defense useful only in late May?
Answer: the model's score_std has a regime-dependent distribution.
When does high score_std mean "crash coming" vs "false alarm"?
"""
import os, json, numpy as np, pandas as pd

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(SCRIPT_DIR, "results", "benchmark_data.parquet")
OUT_PATH  = os.path.join(SCRIPT_DIR, "results", "dispersion_deep_dive.json")

def main():
    rec = pd.read_parquet(DATA_PATH).sort_values("date").reset_index(drop=True)
    n = len(rec)

    csi5d = rec["csi5d"].values
    sc_std = rec["score_std"].values
    t20 = rec["top20_mean"].values
    idx = rec["idx_ret"].values
    ic = rec["ic"].values
    dates = rec["date"].values

    p95 = float(np.percentile(sc_std, 95))
    p10_std = float(np.percentile(sc_std, 10))

    print("=" * 80)
    print("WHY DOES DISPERSION ONLY WORK IN THE TAIL?")
    print("=" * 80)

    # ====== 1. Score_std time series: is there a regime shift? ======
    print("\n[1] SCORE_STD TIME TREND")
    # Split into 5 equal segments
    seg_size = n // 5
    segments = []
    for i in range(5):
        s = i * seg_size
        e = min((i+1) * seg_size, n)
        sub = rec.iloc[s:e]
        triggered = (sub["score_std"] > p95).sum()
        crash_happened = (sub["top20_mean"] < -2).sum()
        segments.append({
            "period": f"{sub['date'].iloc[0]} ~ {sub['date'].iloc[-1]}",
            "n_days": len(sub),
            "score_std_mean": float(sub["score_std"].mean()),
            "score_std_std": float(sub["score_std"].std()),
            "score_std_max": float(sub["score_std"].max()),
            "pct_above_p95": float(triggered / len(sub) * 100),
            "crash_days": int(crash_happened),
            "crash_with_dispersion_trigger": int(((sub["top20_mean"] < -2) & (sub["score_std"] > p95)).sum()),
            "top20_mean": float(sub["top20_mean"].mean()),
        })
        print(f"  {segments[-1]['period']}: sc_std avg={segments[-1]['score_std_mean']:.4f}  "
              f"max={segments[-1]['score_std_max']:.4f}  >P95={segments[-1]['pct_above_p95']:.1f}%  "
              f"crashes={segments[-1]['crash_days']}  caught={segments[-1]['crash_with_dispersion_trigger']}")

    # ====== 2. Every P95 trigger day in detail ======
    print(f"\n[2] EVERY P95 TRIGGER DAY (score_std > {p95:.4f})")
    print(f"  {'Date':<12} {'top20':>7} {'csi5d':>7} {'IC':>7} {'sc_std':>8} {'CS volat':>10} {'WasCrash':>10} {'GoodCall?':>10}")
    print("-" * 80)

    p95_days = rec[rec["score_std"] > p95].sort_values("date")
    good_calls, bad_calls = 0, 0
    for _, r in p95_days.iterrows():
        is_crash = r["top20_mean"] < -2
        is_good = r["top20_mean"] < 0  # any negative day, reducing helped
        good_calls += is_good
        bad_calls += not is_good and is_crash is False
        print(f"  {r['date']:<12} {r['top20_mean']:>+6.2f}% {r['csi5d']:>+6.1f}% "
              f"{r['ic']:>+6.4f} {r['score_std']:>8.4f} {r['csi20vol']:>10.6f} "
              f"{'CRASH' if is_crash else '':>10} {'YES' if is_good else 'no':>10}")

    print(f"\n  Good calls (top20<0, reducing helped): {good_calls}/{len(p95_days)}")
    print(f"  Bad calls (top20>=0, reducing hurt):  {bad_calls}/{len(p95_days)}")

    # ====== 3. The false alarm days — what did they have in common? ======
    false_alarms = rec[(rec["score_std"] > p95) & (rec["top20_mean"] >= 0)]
    print(f"\n[3] FALSE ALARMS: {len(false_alarms)} days where dispersion triggered but market was fine")
    for _, r in false_alarms.iterrows():
        print(f"  {r['date']}: top20={r['top20_mean']:+.2f}%  csi5d={r['csi5d']:+.1f}%  "
              f"csi20vol={r['csi20vol']:.4f}  ic={r['ic']:+.4f}  sc_skew={r['score_skew']:+.3f}")

    # ====== 4. Score_std vs market regime: when does dispersion predict crash? ======
    print(f"\n[4] SCORE_STD * MARKET REGIME — Interaction Analysis")
    # High score_std in bearish momentum vs bullish momentum
    regimes = [
        ("BEAR (csi5d<-2%)", csi5d < -2),
        ("NEUTRAL (csi5d -2~+2%)", (csi5d >= -2) & (csi5d <= 2)),
        ("BULL (csi5d>+2%)", csi5d > 2),
    ]
    for rname, rmask in regimes:
        sub = rec[rmask]
        if len(sub) == 0: continue
        # Within this regime, does high score_std predict worse returns?
        hi_std = sub["score_std"] >= np.percentile(sub["score_std"], 70) if len(sub) >= 5 else np.ones(len(sub), bool)
        lo_std = sub["score_std"] <= np.percentile(sub["score_std"], 30) if len(sub) >= 5 else np.ones(len(sub), bool)
        print(f"  {rname} (n={len(sub):>2}): ")
        print(f"    All:          top20={sub['top20_mean'].mean():+.2f}%")
        if hi_std.sum() > 0:
            print(f"    Hi-sc_std(T30): top20={sub[hi_std]['top20_mean'].mean():+.2f}%  "
                  f"(crashes: {(sub[hi_std]['top20_mean']<-2).sum()}/{hi_std.sum()})")
        if lo_std.sum() > 0:
            print(f"    Lo-sc_std(B30): top20={sub[lo_std]['top20_mean'].mean():+.2f}%  "
                  f"(crashes: {(sub[lo_std]['top20_mean']<-2).sum()}/{lo_std.sum()})")

    # ====== 5. Correlation: score_std vs IC — when model is CONFIDENT, is it RIGHT? ======
    print(f"\n[5] SCORE_STD vs MODEL IC")
    corr_sc_ic = float(np.corrcoef(sc_std, ic)[0, 1])
    print(f"  corr(score_std, IC) = {corr_sc_ic:+.4f}")

    # Split: days when IC is HIGH vs LOW
    hi_ic = ic >= np.percentile(ic, 70)
    lo_ic = ic <= np.percentile(ic, 30)
    print(f"  When IC is HIGH (P70+): score_std avg={sc_std[hi_ic].mean():.4f},  "
          f"top20 avg={t20[hi_ic].mean():+.2f}%")
    print(f"  When IC is LOW  (P30-): score_std avg={sc_std[lo_ic].mean():.4f},  "
          f"top20 avg={t20[lo_ic].mean():+.2f}%")

    # The key question: when score_std is high AND IC is high → model is RIGHT confidently → good
    #                 when score_std is high AND IC is low  → model is WRONG confidently → crash
    for ic_label, ic_mask in [("IC>P70 (model right)", hi_ic), ("IC<P30 (model wrong)", lo_ic)]:
        sub_ic = rec[ic_mask]
        hi_s = sub_ic["score_std"] >= np.percentile(sub_ic["score_std"], 70) if len(sub_ic) > 5 else np.zeros(len(sub_ic), bool)
        if hi_s.sum() > 0:
            print(f"  {ic_label}: when also hi-sc_std → top20={sub_ic[hi_s]['top20_mean'].mean():+.2f}%")

    # ====== 6. WHY it's concentrated: SCORE_STD AUTOCORRELATION ======
    print(f"\n[6] SCORE_STD AUTOCORRELATION (momentum of dispersion itself)")
    for lag in [1, 2, 3, 5, 10]:
        if lag < len(sc_std) - 1:
            ac = float(np.corrcoef(sc_std[:-lag] if lag>0 else sc_std,
                                    sc_std[lag:])[0,1])
            print(f"  Lag-{lag:>2}: r={ac:+.4f}")

    # Clustering: how many sequential days are above P95?
    above = sc_std > p95
    runs = []
    run_start, run_len = -1, 0
    for i, a in enumerate(above):
        if a:
            if run_start < 0: run_start = i
            run_len += 1
        else:
            if run_len > 0:
                runs.append({"start": dates[run_start], "end": dates[i-1], "len": run_len})
                run_len = 0
    if run_len > 0:
        runs.append({"start": dates[run_start], "end": dates[n-1], "len": run_len})

    print(f"\n  Dispersion trigger clusters (score_std > P95 runs):")
    for r in runs:
        seg = rec[(rec["date"] >= r["start"]) & (rec["date"] <= r["end"])]
        print(f"    {r['start']}~{r['end']} ({r['len']}d): top20 avg={seg['top20_mean'].mean():+.2f}%  "
              f"csi5d avg={seg['csi5d'].mean():+.1f}%  IC avg={seg['ic'].mean():+.4f}")

    # ====== 7. 20260403: the ONE crash day NO strategy catches ======
    print(f"\n[7] THE UN-CATCHABLE: 20260403 (crash missed by both momentum and dispersion)")
    row = rec[rec["date"] == "20260403"]
    if len(row) > 0:
        r = row.iloc[0]
        print(f"  date={r['date']}  top20={r['top20_mean']:+.2f}%  csi5d={r['csi5d']:+.1f}%  "
              f"sc_std={r['score_std']:.4f}  sc_skew={r['score_skew']:+.3f}  IC={r['ic']:+.4f}")
        # Where does this day rank on various metrics?
        for col in ["csi5d", "score_std", "score_skew", "csi20vol", "top20_score_range", "ic"]:
            pct = float((rec[col] < r[col]).mean() * 100)
            print(f"    {col:>20} = {r[col]:.4f}  (P{pct:.0f})")
        print(f"    → This day had STRONG bullish momentum (csi5d=+3.5%), model was CONVINCENT (low sc_std),")
        print(f"      market crashed anyway. This is a regime-switch day — un-hedgeable.")

    # ====== 8. Summary: score_std regime shift detection ======
    print(f"\n[8] STRUCTURAL PATTERN")
    # Split at 20260515 (after which crash days cluster)
    early = rec[rec["date"] <= "20260515"]
    late  = rec[rec["date"] > "20260515"]
    for tag, sub in [("EARLY (Feb-Apr)", early), ("LATE (May)", late)]:
        trig = (sub["score_std"] > p95).sum()
        crash = (sub["top20_mean"] < -2).sum()
        trig_crash = ((sub["score_std"] > p95) & (sub["top20_mean"] < -2)).sum()
        print(f"  {tag}: {len(sub)}d  score_std>P95={trig}  crashes={crash}  caught={trig_crash}")
    print(f"  → {int((late['score_std'] > p95).sum())} of {len(p95_days)} P95 triggers happen in late May")

    # ====== Save ======
    output = {
        "p95_threshold": p95,
        "p10_threshold": p10_std,
        "time_segments": segments,
        "trigger_day_detail": p95_days[["date","top20_mean","csi5d","ic","score_std","csi20vol","score_skew"]].to_dict("records"),
        "false_alarm_days": false_alarms[["date","top20_mean","csi5d","ic","score_std","csi20vol","score_skew"]].to_dict("records") if len(false_alarms) > 0 else [],
        "score_std_autocorr": {f"lag{lag}": float(np.corrcoef(sc_std[:-lag], sc_std[lag:])[0,1]) for lag in [1,2,3,5,10]},
        "trigger_clusters": runs,
        "uncatchable_20260403": row[["date","top20_mean","csi5d","score_std","score_skew","ic","csi20vol"]].to_dict("records") if len(row) > 0 else [],
    }
    with open(OUT_PATH, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\n  Saved to {OUT_PATH}")

if __name__ == "__main__":
    main()
