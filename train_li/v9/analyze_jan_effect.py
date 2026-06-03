"""Analyze why January 2026 contributed 85% of total returns (Feb-May near flat).

Hypotheses:
  A. Jan 2026 was a strong bull month → model captured beta, not alpha
  B. Model signal decayed after Jan (train data too old)
  C. Market regime shift after Jan (dispersion/correlation changed)
"""
import numpy as np, pandas as pd, os, sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(SCRIPT_DIR)
SCORES = os.path.join(ROOT, "v7", "results", "daily_scores_spatial_t1.parquet")
PARQUET = os.path.join(ROOT, "processed", "all_data.parquet")
INDEX_P = os.path.join(ROOT, "data", "market", "000300.SH.csv")

# ── CSI300 ──
csi = pd.read_csv(INDEX_P, dtype={"trade_date": str})
csi["pct"] = pd.to_numeric(csi["pct_chg"], errors="coerce") / 100.0
csi_d = dict(zip(csi["trade_date"], csi["pct"]))

# ── scores ──
scores = pd.read_parquet(SCORES)
scores["month"] = scores["trade_date"].str[:6]

# ── all_data (returns only, filter to 2026) ──
df = pd.read_parquet(PARQUET, columns=["trade_date", "ts_code", "pct_chg"])
df["trade_date"] = df["trade_date"].astype(str)
df = df[df["trade_date"] >= "20260101"]
df["month"] = df["trade_date"].str[:6]
df["pct"] = pd.to_numeric(df["pct_chg"], errors="coerce") / 100.0
df.dropna(subset=["pct"], inplace=True)

# ═══════════════════════════════════════════════════════
# 1. CSI300 by month
# ═══════════════════════════════════════════════════════
print("═" * 70)
print(" CSI300 Monthly Performance (2026)")
print("═" * 70)
csi_dates = sorted([d for d in csi_d if d.startswith("2026")])
for m in sorted(set(d[:6] for d in csi_dates)):
    md = [d for d in csi_dates if d.startswith(m)]
    rets = [csi_d[d] for d in md]
    cum = np.prod([1 + r for r in rets]) - 1
    vol = np.std(rets)*np.sqrt(252)*100 if len(rets) > 1 else 0
    sharpe = np.mean(rets)/np.std(rets)*np.sqrt(252) if len(rets) > 1 and np.std(rets) > 0 else 0
    print(f"  {m}: {cum*100:+.2f}%  ({len(md)} days)  μ={np.mean(rets)*100:+.3f}%/d  σ={vol:.1f}%  SR={sharpe:.2f}")

# ═══════════════════════════════════════════════════════
# 2. Score IC by month
# ═══════════════════════════════════════════════════════
print(f"\n{'═'*70}")
print(" Rank IC by Month (v7 scores vs next-day return)")
print("═" * 70)

# Merge scores with next-day returns
df["next_date"] = None
all_dates = sorted(df["trade_date"].unique())
date_map = {d: all_dates[i+1] if i+1 < len(all_dates) else None for i, d in enumerate(all_dates)}
df["next_date"] = df["trade_date"].map(date_map)

# Self-join for next-day returns
next_ret = df[["trade_date", "ts_code", "pct"]].rename(columns={"trade_date": "next_date", "pct": "next_pct"})
merged = scores.merge(df[["trade_date", "ts_code", "pct", "next_date"]], on=["trade_date", "ts_code"], how="inner")
merged = merged.merge(next_ret, on=["next_date", "ts_code"], how="inner")
merged.dropna(subset=["score", "next_pct"], inplace=True)

from scipy.stats import spearmanr
print(f"\n  {'Month':<8} {'N_stocks':>8} {'IC':>8} {'p-val':>8} {'CSI_cum':>8}")
print(f"  {'─'*50}")
monthly_ic = []
for m in sorted(merged["trade_date"].str[:6].unique()):
    mdf = merged[merged["trade_date"].str.startswith(m)]
    if len(mdf) < 30: continue
    ic, pv = spearmanr(mdf["score"], mdf["next_pct"])
    csi_cum = 1.0
    for d in [d for d in csi_dates if d.startswith(m)]:
        csi_cum *= 1 + csi_d.get(d, 0)
    monthly_ic.append((m, ic, pv, len(mdf), csi_cum - 1))
    print(f"  {m:<8} {len(mdf):>8} {ic:>+7.4f} {pv:>8.4f} {csi_cum*100-100:>+7.2f}%")

# ═══════════════════════════════════════════════════════
# 3. Cross-sectional dispersion by month
# ═══════════════════════════════════════════════════════
print(f"\n{'═'*70}")
print(" Cross-Sectional Return Dispersion by Month")
print("═" * 70)
print(f"  {'Month':<8} {'σ_cross(daily)':>15} {'median':>10} {'Q75-Q25':>10}")
print(f"  {'─'*50}")

for m in sorted(df["month"].unique()):
    mdf = df[df["month"] == m]
    daily_std = mdf.groupby("trade_date")["pct"].std()
    print(f"  {m:<8} {daily_std.mean()*100:>14.2f}% {daily_std.median()*100:>9.2f}% "
          f"{(daily_std.quantile(0.75)-daily_std.quantile(0.25))*100:>9.2f}%")

# ═══════════════════════════════════════════════════════
# 4. What % of top-5 stocks come from which sector in Jan vs later?
# ═══════════════════════════════════════════════════════
print(f"\n{'═'*70}")
print(" Top-N Composition Analysis")
print("═" * 70)

# Simulate daily top-5 portfolio (simplified: just top 5 by score each day)
for period_name, date_range in [("Jan 2026", ("20260105", "20260131")),
                                  ("Feb-May 2026", ("20260201", "20260529"))]:
    p_start, p_end = date_range
    p_scores = scores[(scores["trade_date"] >= p_start) & (scores["trade_date"] <= p_end)]
    daily_top5 = []
    for d, sdf in p_scores.groupby("trade_date"):
        top5 = sdf.nlargest(5, "score")
        daily_top5.append(top5)
    if daily_top5:
        all_top5 = pd.concat(daily_top5)
        # Get returns for top-5 stocks
        top5_ret = all_top5.merge(df[["trade_date", "ts_code", "pct"]], on=["trade_date", "ts_code"], how="inner")
        avg_ret = top5_ret["pct"].mean() * 100 if len(top5_ret) > 0 else 0
        n_days = len(daily_top5)
        # Cumulative equal-weight of top-5
        cum_ret = 0
        dret = []
        for d, sdf in top5_ret.groupby("trade_date"):
            dret.append(sdf["pct"].mean())
        cum_ret = np.prod([1+r for r in dret]) - 1 if dret else 0
        print(f"\n  {period_name} ({n_days} days):")
        print(f"    Top-5 avg daily return: {avg_ret:+.3f}%")
        print(f"    Top-5 cumulative: {cum_ret*100:+.2f}%")
        print(f"    Top-5 daily std: {np.std(dret)*100:.2f}%")

# ═══════════════════════════════════════════════════════
# 5. Model IC per 10-day window (same as backtest windows)
# ═══════════════════════════════════════════════════════
print(f"\n{'═'*70}")
print(" IC per 10-Trading-Day Window")
print("═" * 70)
print(f"  {'Window':>6} {'Dates':<22} {'IC':>8} {'N_stocks':>8}")
print(f"  {'─'*55}")

tdates = sorted(merged["trade_date"].unique())
for wi in range(0, len(tdates)-9, 10):
    w_dates = tdates[wi:wi+10]
    wdf = merged[merged["trade_date"].isin(w_dates)]
    if len(wdf) < 100: continue
    ic, _ = spearmanr(wdf["score"], wdf["next_pct"])
    print(f"  {wi//10+1:>6} {w_dates[0]}~{w_dates[-1]:<10} {ic:>+7.4f} {len(wdf):>8}")

# ═══════════════════════════════════════════════════════
# 6. Jan vs Feb-May: model beta to CSI300
# ═══════════════════════════════════════════════════════
print(f"\n{'═'*70}")
print(" Top-5 Portfolio Beta to CSI300")
print("═" * 70)

for period_name, date_range in [("Jan 5-30", ("20260105", "20260130")),
                                  ("Feb 3 - May 29", ("20260203", "20260529"))]:
    p_scores = scores[(scores["trade_date"] >= date_range[0]) & (scores["trade_date"] <= date_range[1])]
    daily_top_ret = []
    daily_csi = []
    for d, sdf in p_scores.groupby("trade_date"):
        if d not in csi_d: continue
        top5 = sdf.nlargest(5, "score")
        top5_r = top5.merge(df[["trade_date", "ts_code", "pct"]], on=["trade_date", "ts_code"], how="inner")
        if len(top5_r) < 5: continue
        daily_top_ret.append(top5_r["pct"].mean())
        daily_csi.append(csi_d[d])
    
    if len(daily_top_ret) > 5:
        top_arr = np.array(daily_top_ret)
        csi_arr = np.array(daily_csi)
        # Beta = cov(top, csi) / var(csi)
        beta = np.cov(top_arr, csi_arr)[0, 1] / np.var(csi_arr)
        # Correlation
        corr = np.corrcoef(top_arr, csi_arr)[0, 1]
        # Alpha = mean(top) - beta * mean(csi)
        alpha = (np.mean(top_arr) - beta * np.mean(csi_arr)) * 100
        # Excess per day
        excess = np.mean(top_arr - csi_arr) * 100
        print(f"\n  {period_name}:")
        print(f"    Days: {len(daily_top_ret)}")
        print(f"    Top-5 mean ret/day: {np.mean(top_arr)*100:+.3f}%")
        print(f"    CSI300 mean ret/day: {np.mean(csi_arr)*100:+.3f}%")
        print(f"    Excess ret/day: {excess:+.3f}%")
        print(f"    Beta: {beta:.2f}")
        print(f"    Correlation: {corr:.2f}")
        print(f"    Daily alpha: {alpha:+.3f}%")
        # What if beta=1?
        hyp_ret_no_alpha = np.mean(top_arr) - alpha/100
        print(f"    If β=1: {hyp_ret_no_alpha*100*len(daily_top_ret):+.2f}% cumulative")
