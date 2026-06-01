"""
Test Strategy B (CSI5d < -1% -> 80% position) on 10 rolling 10-day windows in 2026.
Outputs per-window details with market context and trigger behavior.
"""
import os, json, numpy as np, pandas as pd

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(SCRIPT_DIR, "results", "benchmark_data.parquet")
OUT_PATH  = os.path.join(SCRIPT_DIR, "results", "strategy_b_10windows.json")

MIN_POS = 0.80
WINDOW_SIZE = 10
N_WINDOWS = 10

def segment_stats(sub, tag):
    """Compute strategy metrics for one 10-day segment."""
    tr = sub["top20_mean"].values
    idx = sub["idx_ret"].values
    csi5d = sub["csi5d"].values
    dates = sub["date"].values

    # Baseline
    base_cum = np.prod(1 + tr / 100)
    base_daily = tr

    # Strategy B
    weight = np.where(csi5d < -1.0, MIN_POS, 1.0)
    strat_daily = tr * weight
    strat_cum = np.prod(1 + strat_daily / 100)

    triggered_idx = np.where(weight < 1.0)[0]
    trigger_dates = dates[triggered_idx].tolist()
    trigger_details = []
    for i in triggered_idx:
        trigger_details.append({
            "date": str(dates[i]),
            "csi5d": float(csi5d[i]),
            "top20_raw": float(tr[i]),
            "top20_reduced": float(tr[i] * MIN_POS),
        })

    idx_cum = np.prod(1 + idx / 100)
    excess = strat_cum / idx_cum - 1

    return {
        "segment": tag,
        "start_date": str(dates[0]),
        "end_date": str(dates[-1]),
        "n_days": len(tr),
        "n_triggered": int(len(triggered_idx)),
        "trigger_dates": trigger_dates,
        "trigger_details": trigger_details,
        "baseline_return": float(base_cum - 1),
        "baseline_best_day": float(tr.max()),
        "baseline_worst_day": float(tr.min()),
        "strategy_return": float(strat_cum - 1),
        "strategy_best_day": float(strat_daily.max()),
        "strategy_worst_day": float(strat_daily.min()),
        "csi300_return": float(idx_cum - 1),
        "excess_vs_csi300": float(strat_cum - idx_cum),
        "improvement_vs_baseline": float(strat_cum - base_cum),
        "mean_csi5d": float(csi5d.mean()),
        "mean_csi20vol": float(sub["csi20vol"].mean()),
        "top20_mean_daily": float(tr.mean()),
        "idx_mean_daily": float(idx.mean()),
    }

def main():
    rec = pd.read_parquet(DATA_PATH).sort_values("date").reset_index(drop=True)
    n_total = len(rec)
    print(f"[10-Window Test] Total benchmark days: {n_total}")
    print(f"  Date range: {rec['date'].iloc[0]} ~ {rec['date'].iloc[-1]}")

    # Full period summary for reference
    full_weight = np.where(rec["csi5d"].values < -1.0, MIN_POS, 1.0)
    full_daily = rec["top20_mean"].values * full_weight
    full_cum = np.prod(1 + full_daily / 100)
    base_cum = np.prod(1 + rec["top20_mean"].values / 100)
    print(f"  Full-period baseline: {base_cum-1:+.2%}")
    print(f"  Full-period strategy: {full_cum-1:+.2%} (improvement: {full_cum/base_cum-1:+.2%})")

    # Select 10 evenly-spaced 10-day windows
    total_windows = n_total - WINDOW_SIZE + 1
    step = max(1, total_windows // N_WINDOWS)
    start_indices = [i * step for i in range(N_WINDOWS)]

    windows = []
    for si, idx in enumerate(start_indices):
        end_idx = min(idx + WINDOW_SIZE, n_total)
        sub = rec.iloc[idx:end_idx]
        if len(sub) < 5:
            continue
        w = segment_stats(sub, f"W{si+1}_days{idx+1}-{end_idx}")
        windows.append(w)

    # ---- Aggregate statistics ----
    imp = [w["improvement_vs_baseline"] for w in windows]
    wins = sum(1 for x in imp if x > 0)
    losses = sum(1 for x in imp if x < 0)
    ties = sum(1 for x in imp if x == 0)

    # ---- Per-trigger market outcome analysis ----
    all_triggered = []
    for w in windows:
        for td in w["trigger_details"]:
            all_triggered.append({
                "window": w["segment"],
                **td,
                "csi300_that_day": float(
                    rec.loc[rec["date"] == td["date"], "idx_ret"].values[0]
                    if td["date"] in rec["date"].values else 0
                ),
            })

    # If we DIDN'T reduce position on triggered days, what would have happened?
    trigger_compare = []
    for td in all_triggered:
        d = td["date"]
        row = rec[rec["date"] == d]
        if len(row) > 0:
            trigger_compare.append({
                "date": d,
                "top20_full_return": float(row["top20_mean"].values[0]),
                "top20_reduced_return": float(row["top20_mean"].values[0] * MIN_POS),
                "csi300_return": float(row["idx_ret"].values[0]),
                "csi5d": float(row["csi5d"].values[0]),
                "was_crash_day": bool(row["top20_mean"].values[0] < -2),
                "was_up_day": bool(row["top20_mean"].values[0] > 1),
            })

    # ---- Output ----
    output = {
        "strategy": "B: CSI5d < -1% risk-off to 80%",
        "test_description": f"10 rolling {WINDOW_SIZE}-day windows evenly spread across 2026 test period",
        "full_period": {
            "start": str(rec["date"].iloc[0]),
            "end": str(rec["date"].iloc[-1]),
            "total_days": int(n_total),
            "baseline_return": float(base_cum - 1),
            "strategy_return": float(full_cum - 1),
            "improvement": float(full_cum / base_cum - 1) if base_cum > 0 else 0,
        },
        "windows": windows,
        "cross_window_summary": {
            "n_windows": len(windows),
            "mean_improvement": float(np.mean(imp)),
            "std_improvement": float(np.std(imp)),
            "wins": int(wins),
            "losses": int(losses),
            "ties": int(ties),
            "win_rate": f"{wins}/{len(windows)} ({wins/len(windows)*100:.0f}%)",
            "improvements": [float(x) for x in imp],
        },
        "trigger_impact_analysis": {
            "total_triggered_days": len(all_triggered),
            "triggered_days_detail": all_triggered,
            "crash_days_saved": trigger_compare,
            "saved_count": sum(1 for x in trigger_compare if x["was_crash_day"]),
            "false_alarm_count": sum(1 for x in trigger_compare if x["was_up_day"]),
            "neutral_count": sum(1 for x in trigger_compare if not x["was_crash_day"] and not x["was_up_day"]),
        },
        "config": {
            "min_position": MIN_POS,
            "window_size_days": WINDOW_SIZE,
            "n_windows": N_WINDOWS,
            "trigger_rule": "csi5d < -1.0%",
        },
    }

    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    with open(OUT_PATH, "w") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    print(f"\n  Results saved to {OUT_PATH}")

    # ---- Print summary table ----
    print(f"\n{'='*100}")
    print(f"10-WINDOW TEST RESULTS")
    print(f"{'='*100}")
    print(f"{'Window':>10} {'Dates':<24} {'Trig':>4} {'Baseline':>8} {'Strategy':>8} {'CSI300':>8} {'Improve':>8} {'Excess':>8}")
    print("-"*100)
    for w in windows:
        print(f"{w['segment']:>10} {w['start_date']}~{w['end_date']}  "
              f"{w['n_triggered']:>3}d {w['baseline_return']:>+7.2%} {w['strategy_return']:>+7.2%} "
              f"{w['csi300_return']:>+7.2%} {w['improvement_vs_baseline']:>+7.2%} {w['excess_vs_csi300']:>+7.2%}")
    print("-"*100)
    print(f"{'AVERAGE':>10} {'':<24} {'':>4} {'':>8} {'':>8} {'':>8} "
          f"{np.mean(imp):>+7.2%} {'':>8}")
    print(f"  Wins: {wins}/{len(windows)}  (significantly positive: {sum(1 for x in imp if x>0.002)}/{len(windows)})")

    # Crash day analysis
    if trigger_compare:
        crashes = [x for x in trigger_compare if x["was_crash_day"]]
        false_alarms = [x for x in trigger_compare if x["was_up_day"]]
        print(f"\n  Trigger accuracy: {len(crashes)} crash saved / {len(false_alarms)} false alarms / "
              f"{len(trigger_compare)-len(crashes)-len(false_alarms)} neutral ({len(trigger_compare)} total triggers)")

if __name__ == "__main__":
    main()
