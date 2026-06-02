"""diagnose_gap_v2.py — Fast gap diagnostics using filtered parquet loading."""
import pandas as pd, numpy as np
from scipy.stats import spearmanr

SCORES_PATH = r"D:\Workspace\DL_HW\deep-learning2026final-lab\train_li\v6\results\daily_scores.parquet"
PARQUET_PATH = r"D:\Workspace\DL_HW\deep-learning2026final-lab\train_li\processed\all_data.parquet"

scores = pd.read_parquet(SCORES_PATH)
dates_s = sorted(scores["trade_date"].unique())
print(f"Scores: {len(dates_s)} dates, {len(scores):,} rows")

# Only load needed dates
needed = set(dates_s)
for i, d in enumerate(dates_s):
    for offset in range(1, 6):
        if i + offset < len(dates_s):
            needed.add(dates_s[i + offset])
print(f"Loading {len(dates_s)} score dates + {len(needed)-len(dates_s)} return dates")

df = pd.read_parquet(PARQUET_PATH)
df["trade_date"] = df["trade_date"].astype(str)
df = df[df["trade_date"].isin(needed)].copy()
print(f"Loaded {len(df):,} rows, {df['ts_code'].nunique()} stocks")

# Build fast lookup: dict[date][ts_code] = dict of fields
cols_needed = {"pct_chg", "total_mv", "vol", "turnover_rate"}
ret_d, mv_d = {}, {}
for d in sorted(needed):
    sub = df[df["trade_date"] == d]
    ret_d[d] = dict(zip(sub["ts_code"], sub["pct_chg"].astype(float)))
    mv_d[d] = dict(zip(sub["ts_code"], sub["total_mv"].astype(float)))

# Score lookup
score_d = {}
for _, row in scores.iterrows():
    score_d.setdefault(row["trade_date"], {})[row["ts_code"]] = row["score"]
del df

# =========== DIAG 1: Monthly IC ===========
print(f"\n{'='*60}")
print(" DIAG 1: IC by Month + 5-day window")
print(f"{'='*60}")

daily_ics = []
for i in range(len(dates_s) - 1):
    fd, rd = dates_s[i], dates_s[i + 1]
    if fd not in score_d or rd not in ret_d: continue
    ss = score_d[fd]; rr = ret_d[rd]
    common = [ts for ts in ss if ts in rr]
    if len(common) < 100: continue
    s_vals = [ss[ts] for ts in common]
    r_vals = [rr[ts] for ts in common]
    ic = spearmanr(s_vals, r_vals).correlation
    if not np.isnan(ic):
        daily_ics.append((fd, ic))

if not daily_ics:
    print("  ERROR: no IC computed")
else:
    dates_only = [x[0] for x in daily_ics]
    ics = [x[1] for x in daily_ics]
    
    # By month
    monthly = {}
    for d, ic in daily_ics:
        m = d[:7]
        monthly.setdefault(m, []).append(ic)
    print(f"  {'Month':<10} {'Mean IC':>8} {'Std':>8} {'IC>0':>7} {'N':>5}")
    for m in sorted(monthly.keys()):
        v = monthly[m]
        print(f"  {m:<10} {np.mean(v):>+7.4f} {np.std(v):>8.4f} {np.mean([x>0 for x in v]):>6.1%} {len(v):>5}")
    
    # By 5-day window
    print(f"\n  --- 5-day windows ---")
    for w in range(0, len(daily_ics), 5):
        if w + 3 > len(daily_ics): continue
        w_ics = ics[w:w+5]
        print(f"  {dates_only[w]}~{dates_only[min(w+4,len(dates_only)-1)]}: "
              f"IC={np.mean(w_ics):+.4f} >0={np.mean([x>0 for x in w_ics]):.0%}")
    
    print(f"\n  Overall: IC={np.mean(ics):+.4f} std={np.std(ics):.4f} >0={np.mean([x>0 for x in ics]):.1%}")

# =========== DIAG 2: T+1 vs T+5 labels ===========
print(f"\n{'='*60}")
print(" DIAG 2: T+1 vs T+5 label IC")
print(f"{'='*60}")

t1_ics, t5_ics = [], []
for i in range(len(dates_s) - 6):
    fd = dates_s[i]
    if fd not in score_d: continue
    ss = score_d[fd]
    
    # T+1
    t1d = dates_s[i + 1]
    if t1d in ret_d:
        common = [(ts, ss[ts], ret_d[t1d][ts]) for ts in ss if ts in ret_d[t1d]]
        if len(common) > 50:
            s, r = [x[1] for x in common], [x[2] for x in common]
            ic = spearmanr(s, r).correlation
            if not np.isnan(ic): t1_ics.append(ic)
    
    # T+5 cumulative
    rd_dates = [dates_s[i + o] for o in range(1, 6) if i + o < len(dates_s)]
    if len(rd_dates) < 5: continue
    valid_ts = set(ss.keys())
    for rd in rd_dates:
        if rd in ret_d:
            valid_ts &= set(ret_d[rd].keys())
    if len(valid_ts) < 50: continue
    
    s_vals, r_vals = [], []
    for ts in valid_ts:
        cum = sum(ret_d[rd][ts] for rd in rd_dates if ts in ret_d[rd])
        s_vals.append(ss[ts])
        r_vals.append(cum)
    ic5 = spearmanr(s_vals, r_vals).correlation
    if not np.isnan(ic5): t5_ics.append(ic5)

print(f"  IC vs T+1:  {np.mean(t1_ics):+.4f}  std={np.std(t1_ics):.4f}  >0={np.mean([x>0 for x in t1_ics]):.1%}")
print(f"  IC vs T+5:  {np.mean(t5_ics):+.4f}  std={np.std(t5_ics):.4f}  >0={np.mean([x>0 for x in t5_ics]):.1%}")
if t1_ics and t5_ics:
    print(f"  T+5 IC / T+1 IC: {np.mean(t5_ics)/max(np.mean(t1_ics), 0.001):.2f}x")

# =========== DIAG 3: IC by market cap ===========
print(f"\n{'='*60}")
print(" DIAG 3: IC by Market Cap Tercile")
print(f"{'='*60}")

cap_buckets = {"top 1/3": [], "mid 1/3": [], "bot 1/3": []}
for i in range(len(dates_s) - 1):
    fd, rd = dates_s[i], dates_s[i + 1]
    if fd not in score_d or fd not in mv_d or rd not in ret_d: continue
    ss = score_d[fd]; rr = ret_d[rd]; mm = mv_d[fd]
    
    items = [(ts, ss[ts], mm[ts]) for ts in ss if ts in rr and ts in mm and mm[ts] > 0]
    if len(items) < 150: continue
    items.sort(key=lambda x: x[2], reverse=True)
    N = len(items)
    
    bounds = [(0, N//3), (N//3, 2*N//3), (2*N//3, N)]
    for (bname, (s, e)), bucket in zip(
        [("top 1/3", (0, N//3)), ("mid 1/3", (N//3, 2*N//3)), ("bot 1/3", (2*N//3, N))],
        [items[0:N//3], items[N//3:2*N//3], items[2*N//3:N]]
    ):
        if len(bucket) < 30: continue
        s_vals = [x[1] for x in bucket]
        r_vals = [rr[x[0]] for x in bucket if x[0] in rr]
        if len(r_vals) < 20: continue
        s_vals = s_vals[:len(r_vals)]
        ic = spearmanr(s_vals, r_vals).correlation
        if not np.isnan(ic): cap_buckets[bname].append(ic)

for bname in ["top 1/3", "mid 1/3", "bot 1/3"]:
    vals = cap_buckets[bname]
    if vals:
        print(f"  {bname:>10}: IC={np.mean(vals):+.4f} std={np.std(vals):.4f} >0={np.mean([x>0 for x in vals]):.1%}")

# =========== DIAG 4: Feature shift 2025 vs 2026 ===========
print(f"\n{'='*60}")
print(" DIAG 4: Feature Distribution 2025 vs 2026")
print(f"{'='*60}")

# Load both periods separately
for period, start, end in [("2025 val", "20250102", "20251231"), ("2026 test", "20260203", "20260529")]:
    sub = pd.read_parquet(PARQUET_PATH)
    sub["trade_date"] = sub["trade_date"].astype(str)
    sub = sub[(sub["trade_date"] >= start) & (sub["trade_date"] <= end)]
    
    print(f"\n  {period} ({start}~{end}, {len(sub):,} rows):")
    for col in ["pct_chg", "turnover_rate"]:
        v = sub[col].astype(float).dropna()
        if len(v) > 0:
            print(f"    {col:>16}: μ={v.mean():+.3f} σ={v.std():.3f} skew={v.skew():+.2f}")
    
    # Daily cross-sectional std
    daily_std = sub.groupby("trade_date")["pct_chg"].std().dropna()
    print(f"    avg cross-sect σ of pct_chg: {daily_std.mean():.2f}%")
    
    # IC between pct_chg and market cap (to test if market favors cap)
    cap_corr = []
    for d in sorted(sub["trade_date"].unique()):
        ss = sub[sub["trade_date"] == d]
        if len(ss) < 100: continue
        p = ss["pct_chg"].astype(float)
        m = ss["total_mv"].astype(float)
        valid = p.notna() & m.notna() & (m > 0)
        if valid.sum() < 100: continue
        corr = spearmanr(p[valid], m[valid]).correlation
        if not np.isnan(corr): cap_corr.append(corr)
    if cap_corr:
        print(f"    RankIC(pct_chg, market_cap): {np.mean(cap_corr):+.4f}")

print(f"\n{'='*60}")
print(" Done.")
print(f"{'='*60}")
