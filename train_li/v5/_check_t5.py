"""Empirical check: how similar are T+1 and T+5 cross-sectional rankings?"""
import numpy as np, pandas as pd
from scipy.stats import spearmanr

df = pd.read_parquet(r"D:\Workspace\DL_HW\LAB5\processed\all_data.parquet")
df["trade_date"] = df["trade_date"].astype(str)
dates_all = sorted(df["trade_date"].unique())

results = []
H = 5  # T+5 label horizon

for di, date in enumerate(dates_all[:-H]):
    t1_ret = []
    t5_ret = []
    
    today = df[df["trade_date"] == date]
    for _, row in today.iterrows():
        ts = row["ts_code"]
        sdf = df[(df["ts_code"] == ts) & (df["trade_date"] > date)].sort_values("trade_date")
        if len(sdf) < H: continue
        t1_ret.append(sdf["pct_chg"].iloc[0])           # next day
        t5_ret.append(sdf["pct_chg"].iloc[:H].sum())    # next 5 days cumulative
    
    if len(t1_ret) < 30: continue
    corr = spearmanr(t1_ret, t5_ret).correlation
    results.append(corr)

    if (di + 1) % 500 == 0:
        print(f"  {di+1} days processed, mean corr={np.mean(results):.4f}", flush=True)

print(f"\n  N dates: {len(results)}")
print(f"  Mean Spearman corr(T+1, T+5): {np.mean(results):.4f}")
print(f"  Std: {np.std(results):.4f}")
print(f"  Corr > 0.7: {np.mean([r>0.7 for r in results]):.1%}")
print(f"  Corr > 0.5: {np.mean([r>0.5 for r in results]):.1%}")
print(f"  Corr < 0.3: {np.mean([r<0.3 for r in results]):.1%}")

# Also compare: which label gives higher rank IC if we use the SAME simple signal?
print("\n  --- T+1 vs T+5 label IC comparison ---")
# Use a simple signal: 5-day momentum (close[t]/close[t-5]-1)
mom_ic1 = []
mom_ic5 = []
for di, date in enumerate(dates_all[:-H]):
    if di < 20: continue  # need at least 20 days of history for momentum
    mom = []
    t1 = []
    t5 = []
    today = df[df["trade_date"] == date]
    for _, row in today.iterrows():
        ts = row["ts_code"]
        sdf = df[(df["ts_code"] == ts) & (df["trade_date"] <= date)].sort_values("trade_date")
        if len(sdf) < 6: continue
        m = sdf["close"].iloc[-1] / sdf["close"].iloc[-6] - 1  # 5-day momentum
        
        fwd = df[(df["ts_code"] == ts) & (df["trade_date"] > date)].sort_values("trade_date")
        if len(fwd) < H: continue
        
        mom.append(m)
        t1.append(fwd["pct_chg"].iloc[0])
        t5.append(fwd["pct_chg"].iloc[:H].sum())
    
    if len(mom) < 30: continue
    ic1 = spearmanr(mom, t1).correlation
    ic5 = spearmanr(mom, t5).correlation
    mom_ic1.append(ic1)
    mom_ic5.append(ic5)

print(f"  5-day momentum → T+1 IC: {np.mean(mom_ic1):+.4f} ± {np.std(mom_ic1):.4f}")
print(f"  5-day momentum → T+5 IC: {np.mean(mom_ic5):+.4f} ± {np.std(mom_ic5):.4f}")
print(f"  T+5 label IC improvement: {np.mean(mom_ic5)-np.mean(mom_ic1):+.4f}")
