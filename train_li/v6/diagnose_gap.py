"""diagnose_gap.py — Diagnose the 55% val→test IC gap.

1. Monthly/weekly IC decomposition
2. T+1 vs T+5 label difficulty comparison
3. IC by market cap bucket
4. Feature distribution shift (2025 vs 2026)
"""
import pandas as pd, numpy as np
from scipy.stats import spearmanr

SCORES_PATH = r"D:\Workspace\DL_HW\deep-learning2026final-lab\train_li\v6\results\daily_scores.parquet"
PARQUET_PATH = r"D:\Workspace\DL_HW\deep-learning2026final-lab\train_li\processed\all_data.parquet"

# ===== Load data =====
scores = pd.read_parquet(SCORES_PATH)
dates_s = sorted(scores["trade_date"].unique())

# Load returns for test period + 5 days forward look
end_date = dates_s[-1]
needed = set(dates_s)
for i, d in enumerate(dates_s):
    for offset in range(1, 6):
        if i + offset < len(dates_s):
            needed.add(dates_s[i + offset])

print(f"[load] {len(needed)} dates needed")

df = pd.read_parquet(PARQUET_PATH)
df["trade_date"] = df["trade_date"].astype(str)

# Stock-level returns lookup
all_stocks = sorted(df["ts_code"].unique())

# Build sorted return arrays per stock
print("[build] stock return series ...")
stock_rets = {}
for ts, sdf in df.groupby("ts_code"):
    sdf = sdf.sort_values("trade_date")
    stock_rets[ts] = dict(zip(sdf["trade_date"], sdf["pct_chg"].astype(float)))

# Also market cap
stock_mv = {}
for ts, sdf in df.groupby("ts_code"):
    sdf = sdf.sort_values("trade_date")
    stock_mv[ts] = dict(zip(sdf["trade_date"], sdf["total_mv"].astype(float)))

del df

# ================================================================
# DIAG 1: Monthly/Weekly IC decomposition
# ================================================================
print(f"\n{'='*60}")
print(" DIAG 1: IC by Month (2026 Feb-May)")
print(f"{'='*60}")

score_d = {}
for _, row in scores.iterrows():
    score_d.setdefault(row["trade_date"], {})[row["ts_code"]] = row["score"]

monthly_ics = {}
all_ics = []
for i in range(len(dates_s) - 1):
    fd, rd = dates_s[i], dates_s[i + 1]
    if fd not in score_d or rd not in stock_rets:
        continue
    
    ss = score_d[fd]
    rr = {ts: stock_rets[ts].get(rd) for ts in ss if ts in stock_rets and rd in stock_rets[ts]}
    
    s_list, r_list = [], []
    for ts, r in rr.items():
        if not np.isnan(r):
            s_list.append(ss[ts])
            r_list.append(r)
    
    if len(s_list) < 50:
        continue
    
    ic = spearmanr(s_list, r_list).correlation
    if not np.isnan(ic):
        month = fd[:6]
        monthly_ics.setdefault(month, []).append(ic)
        all_ics.append(ic)

print(f"{'Month':<10} {'Mean IC':>8} {'Std':>8} {'IC>0':>7} {'Days':>5}")
for m in sorted(monthly_ics.keys()):
    vals = monthly_ics[m]
    print(f"{m:<10} {np.mean(vals):>+7.4f} {np.std(vals):>8.4f} {np.mean([v>0 for v in vals]):>6.1%} {len(vals):>5}")

# Weekly
print(f"\n--- By 5-day rolling window ---")
for w_start in range(0, len(all_ics), 5):
    w_end = min(w_start + 5, len(all_ics))
    if w_end - w_start < 3: continue
    vals = all_ics[w_start:w_end]
    dates_w = dates_s[w_start:w_end]
    print(f"  {dates_w[0]}~{dates_w[-1]}: mean_ic={np.mean(vals):+.4f} "
          f">0={np.mean([v>0 for v in vals]):.0%} n={len(vals)}")

print(f"\n  Overall mean IC: {np.mean(all_ics):+.4f}  std: {np.std(all_ics):.4f}  >0: {np.mean([v>0 for v in all_ics]):.1%}")

# ================================================================
# DIAG 2: T+1 vs T+5 label prediction difficulty
# ================================================================
print(f"\n{'='*60}")
print(" DIAG 2: T+1 vs T+3 vs T+5 return dispersion")
print(f"{'='*60}")

t1_stds, t3_stds, t5_stds = [], [], []

for i in range(len(dates_s) - 6):
    fd = dates_s[i]
    t1_date = dates_s[i + 1]
    t3_date = dates_s[i + 3] if i + 3 < len(dates_s) else dates_s[-1]
    t5_date = dates_s[i + 5] if i + 5 < len(dates_s) else dates_s[-1]
    
    # T+1 return dispersion
    r1 = []
    for ts in stock_rets:
        if fd in stock_rets[ts] and t1_date in stock_rets[ts]:
            r1.append(stock_rets[ts][t1_date])
    if len(r1) > 100:
        t1_stds.append(np.std(r1))
    
    # T+3 cumulative return dispersion
    r3 = []
    for ts in stock_rets:
        vals = []
        for offset in range(1, 4):
            d = dates_s[i + offset] if i + offset < len(dates_s) else None
            if d and fd in stock_rets[ts] and d in stock_rets[ts]:
                vals.append(stock_rets[ts][d])
        if len(vals) == 3:
            r3.append(np.sum(vals))
    if len(r3) > 100:
        t3_stds.append(np.std(r3))
    
    # T+5 cumulative return dispersion
    r5 = []
    for ts in stock_rets:
        vals = []
        for offset in range(1, 6):
            d = dates_s[i + offset] if i + offset < len(dates_s) else None
            if d and fd in stock_rets[ts] and d in stock_rets[ts]:
                vals.append(stock_rets[ts][d])
        if len(vals) == 5:
            r5.append(np.sum(vals))
    if len(r5) > 100:
        t5_stds.append(np.std(r5))

print(f"  T+1  return cross-sectional std: mean={np.mean(t1_stds)*100:.2f}%")
print(f"  T+3  return cross-sectional std: mean={np.mean(t3_stds)*100:.2f}%")
print(f"  T+5  return cross-sectional std: mean={np.mean(t5_stds)*100:.2f}%")
if t1_stds and t5_stds:
    print(f"  T+5 / T+1 std ratio: {np.mean(t5_stds)/np.mean(t1_stds):.2f}x")
    print(f"  → T+5 returns are {np.mean(t5_stds)/np.mean(t1_stds):.1f}x more dispersed → easier to rank")

# Also compute IC of scores vs T+5 returns
print(f"\n  IC of v6 scores vs T+1 returns: already computed above = {np.mean(all_ics):+.4f}")
t5_ics = []
for i in range(len(dates_s) - 6):
    fd = dates_s[i]
    if fd not in score_d: continue
    
    ss = score_d[fd]
    t5_date = dates_s[i + 5]
    
    s_list, r_list = [], []
    for ts in ss:
        vals = []
        for offset in range(1, 6):
            d = dates_s[i + offset] if i + offset < len(dates_s) else None
            if d and ts in stock_rets and d in stock_rets[ts]:
                vals.append(stock_rets[ts][d])
        if len(vals) == 5:
            s_list.append(ss[ts])
            r_list.append(np.sum(vals))
    
    if len(s_list) > 50:
        ic = spearmanr(s_list, r_list).correlation
        if not np.isnan(ic): t5_ics.append(ic)

if t5_ics:
    print(f"  IC of v6 scores vs T+5 returns: {np.mean(t5_ics):+.4f} (std={np.std(t5_ics):.4f}, >0={np.mean([v>0 for v in t5_ics]):.1%})")

# ================================================================
# DIAG 3: IC by market cap bucket
# ================================================================
print(f"\n{'='*60}")
print(" DIAG 3: IC by Market Cap Bucket")
print(f"{'='*60}")

cap_ics = {"large": [], "mid": [], "small": []}

for i in range(len(dates_s) - 1):
    fd, rd = dates_s[i], dates_s[i + 1]
    if fd not in score_d or fd not in stock_mv: continue
    
    ss = score_d[fd]
    mv = stock_mv[fd]
    
    # Filter stocks with MV data
    valid = [(ts, ss[ts], mv[ts]) for ts in ss if ts in mv and mv[ts] > 0]
    if len(valid) < 100: continue
    valid.sort(key=lambda x: x[2], reverse=True)
    N = len(valid)
    
    for bucket_name, (start, end) in [("large", (0, N//3)), ("mid", (N//3, 2*N//3)), ("small", (2*N//3, N))]:
        bucket = valid[start:end] if end <= N else valid[start:]
        if len(bucket) < 20: continue
        
        s_list = [x[1] for x in bucket]
        r_list = []
        for ts, _, _ in bucket:
            if ts in stock_rets and rd in stock_rets[ts]:
                r_list.append(stock_rets[ts][rd])
        
        if len(r_list) < 20: continue
        ic = spearmanr(s_list[:len(r_list)], r_list).correlation
        if not np.isnan(ic):
            cap_ics[bucket_name].append(ic)

for bucket in ["large", "mid", "small"]:
    vals = cap_ics[bucket]
    if vals:
        print(f"  {bucket:>6} cap: IC={np.mean(vals):+.4f} std={np.std(vals):.4f} >0={np.mean([v>0 for v in vals]):.1%} days={len(vals)}")

# ================================================================
# DIAG 4: Feature distribution shift 2025 vs 2026
# ================================================================
print(f"\n{'='*60}")
print(" DIAG 4: Feature Distribution Shift (2025 vs 2026)")
print(f"{'='*60}")

# Load raw parquet for feature analysis
df_full = pd.read_parquet(PARQUET_PATH)
df_full["trade_date"] = df_full["trade_date"].astype(str)

periods = {
    "2025 val":  ("20250102", "20251231"),
    "2026 test": ("20260203", "20260529"),
}

for period_name, (start, end) in periods.items():
    sub = df_full[(df_full["trade_date"] >= start) & (df_full["trade_date"] <= end)]
    print(f"\n  {period_name} ({start}~{end}, {len(sub):,} rows):")
    
    for col in ["pct_chg", "vol", "turnover_rate"]:
        vals = sub[col].astype(float).dropna()
        if len(vals) > 0:
            print(f"    {col:>16}: mean={vals.mean():+8.4f} std={vals.std():8.4f} "
                  f"skew={vals.skew():+7.2f} kurt={vals.kurtosis():+7.2f}")
    
    # Cross-sectional dispersion (avg daily std)
    if "pct_chg" in sub.columns:
        daily_stds = sub.groupby("trade_date")["pct_chg"].apply(lambda x: x.std() if len(x) > 50 else np.nan).dropna()
        if len(daily_stds) > 0:
            print(f"    avg daily cross-sectional std of pct_chg: {daily_stds.mean()*100:.2f}%")

print(f"\n{'='*60}")
print(" Done.")
print(f"{'='*60}")
