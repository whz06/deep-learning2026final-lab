"""Fast empirical check: T+1 vs T+5 label ranking alignment (sampled)."""
import numpy as np, pandas as pd
from scipy.stats import spearmanr

df = pd.read_parquet(r"D:\Workspace\DL_HW\LAB5\processed\all_data.parquet")
df["trade_date"] = df["trade_date"].astype(str)
dates_all = sorted(df["trade_date"].unique())

np.random.seed(42)
sample_stocks = sorted(np.random.choice(sorted(df["ts_code"].unique()), 500, replace=False))
sample_dates = dates_all[500:1500]  # middle dates with sufficient history

H = 5
corrs = []

for di, date in enumerate(sample_dates):
    if date > dates_all[-H-1]: break
    
    t1, t5 = [], []
    for ts in sample_stocks:
        sdf = df[(df["ts_code"] == ts)].sort_values("trade_date")
        pos = sdf["trade_date"].searchsorted(date)
        if pos >= len(sdf): continue
        if sdf["trade_date"].iloc[pos] != date: continue
        if pos + H >= len(sdf): continue
        
        t1.append(sdf["pct_chg"].iloc[pos + 1])
        t5.append(sdf["pct_chg"].iloc[pos+1 : pos+1+H].sum())
    
    if len(t1) < 30: continue
    corrs.append(spearmanr(t1, t5).correlation)
    
    if (di+1) % 200 == 0:
        print(f"  [{di+1}/{len(sample_dates)}] mean_corr={np.mean(corrs):.4f}", flush=True)

print(f"\n=== T+1 vs T+5 Cross-Sectional Rank Correlation ===")
print(f"  N dates: {len(corrs)}")
print(f"  Mean: {np.mean(corrs):.4f}")
print(f"  Std:  {np.std(corrs):.4f}")
print(f"  > 0.7: {np.mean([c>0.7 for c in corrs]):.0%}")
print(f"  > 0.5: {np.mean([c>0.5 for c in corrs]):.0%}")
print(f"  < 0.3: {np.mean([c<0.3 for c in corrs]):.0%}")

# KEY TEST: does a simple factor predict T+5 better than T+1?
print(f"\n=== Factor IC: T+1 label vs T+5 label ===")
for lookback in [5, 10, 20, 60]:
    ic1_vals, ic5_vals = [], []
    for date in sample_dates:
        if date > dates_all[-H-1]: break
        mom, t1, t5 = [], [], []
        for ts in sample_stocks:
            sdf = df[(df["ts_code"] == ts)].sort_values("trade_date")
            pos = sdf["trade_date"].searchsorted(date)
            if pos < lookback + 1 or pos + H >= len(sdf): continue
            if sdf["trade_date"].iloc[pos] != date: continue
            
            m = sdf["close"].iloc[pos] / sdf["close"].iloc[pos - lookback] - 1
            mom.append(m)
            t1.append(sdf["pct_chg"].iloc[pos + 1])
            t5.append(sdf["pct_chg"].iloc[pos+1 : pos+1+H].sum())
        
        if len(mom) < 30: continue
        ic1_vals.append(spearmanr(mom, t1).correlation)
        ic5_vals.append(spearmanr(mom, t5).correlation)
    
    print(f"  mom_{lookback:>2d}d -> T+1 IC: {np.mean(ic1_vals):+.4f} ± {np.std(ic1_vals):.4f}")
    print(f"  mom_{lookback:>2d}d -> T+5 IC: {np.mean(ic5_vals):+.4f} ± {np.std(ic5_vals):.4f}")
    print(f"           delta: {np.mean(ic5_vals)-np.mean(ic1_vals):+.4f} ({np.mean([i5>i1 for i1,i5 in zip(ic1_vals, ic5_vals)]):.0%} days T+5 better)")
