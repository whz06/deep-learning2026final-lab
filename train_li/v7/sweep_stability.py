"""10x10 stability backtest for v7 Spatial, varying N (K=3 fixed)."""
import pandas as pd, numpy as np

SD = r"D:\Workspace\DL_HW\deep-learning2026final-lab\train_li\v7\results\daily_scores_spatial_t1.parquet"
PARQUET = r"D:\Workspace\DL_HW\deep-learning2026final-lab\train_li\processed\all_data.parquet"
INDEX_P = r"D:\Workspace\DL_HW\deep-learning2026final-lab\train_li\data\market\000300.SH.csv"

scores = pd.read_parquet(SD); dates_s = sorted(scores["trade_date"].unique())
needed = set(dates_s)
for i,d in enumerate(dates_s):
    for off in range(1,11):  # up to 10 days forward
        if i+off < len(dates_s): needed.add(dates_s[i+off])

df = pd.read_parquet(PARQUET); df["trade_date"] = df["trade_date"].astype(str)
df = df[df["trade_date"].isin(needed)]
ret_d = {}
for d in sorted(needed):
    sub = df[df["trade_date"]==d]
    ret_d[d] = dict(zip(sub["ts_code"], sub["pct_chg"].astype(float)))
del df

csi = pd.read_csv(INDEX_P, dtype={"trade_date":str})
csi_map = dict(zip(csi["trade_date"], csi["pct_chg"].astype(float)))
csi_dates = sorted(csi_map.keys())
def get_csi5d(date):
    if date not in csi_dates: return 0.0
    idx=csi_dates.index(date); return sum(csi_map[csi_dates[i]] for i in range(max(0,idx-4),idx+1))

sd={}
for _, row in scores.iterrows():
    sd.setdefault(row["trade_date"],{})[row["ts_code"]]=row["score"]
test_list = sorted(sd.keys())

# Same 10 random windows as previous backtests
np.random.seed(42)
valid_starts = [d for d in test_list if test_list.index(d) < len(test_list)-11]
starts = sorted(np.random.choice(valid_starts, 10, replace=False))

K = 3

print(f"{'N':>3}  {'Win1':>6} {'Win2':>6} {'Win3':>6} {'Win4':>6} {'Win5':>6} "
      f"{'Win6':>6} {'Win7':>6} {'Win8':>6} {'Win9':>6} {'Win10':>6} "
      f"{'Mean':>7} {'Std':>6} {'Wins':>5} {'MaxLoss':>7}")
print("-" * 115)

summary = []
for N in [5,6,7,8,9,10]:
    window_results = []
    for wi, start in enumerate(starts):
        si = test_list.index(start)
        holdings = None; daily_p = []
        for t in range(10):
            fd = test_list[si+t]
            rd = test_list[si+t+1] if si+t+1 < len(test_list) else fd
            if fd not in sd or rd not in ret_d: continue
            ss = sd[fd]; rr = ret_d[rd]
            top = [c for c,_ in sorted(ss.items(), key=lambda x:x[1], reverse=True) if c in rr]
            if len(top) < N: continue
            csi5d = get_csi5d(fd)
            pos = 0.80 if csi5d < -1.0 else 1.0
            target = max(1, int(round(N*pos)))
            if holdings is None:
                holdings = top[:target]
            else:
                held_s = [(c, ss.get(c,-1e6)) for c in holdings if c in ss]
                held_s.sort(key=lambda x:x[1], reverse=True)
                to_sell = set(c for c,_ in held_s[-K:]) if len(held_s) > K else set()
                extra = len(holdings) - target
                if extra > 0:
                    cand = [c for c,_ in held_s if c not in to_sell]
                    for c in cand[-extra:]: to_sell.add(c)
                holdings = [c for c in holdings if c not in to_sell]
                held_set = set(holdings)
                for c in top:
                    if len(holdings) >= target: break
                    if c not in held_set: holdings.append(c)
            pr = np.mean([rr.get(c,0.0) for c in holdings]) if holdings else 0.0
            daily_p.append(pr)
        if daily_p: window_results.append(np.sum(daily_p))
    
    if window_results:
        vals = np.array(window_results)
        wins = sum(v > 0 for v in vals)
        mean_v = np.mean(vals); std_v = np.std(vals); max_loss = np.min(vals)
        line = f"{N:>3}  " + "  ".join(f"{v:+5.2f}%" for v in vals) + f"  {mean_v:+6.2f}% {std_v:5.2f}% {wins:>4}/10 {max_loss:+6.2f}%"
        print(line)
        summary.append({"N":N,"mean":round(mean_v,2),"std":round(std_v,2),"wins":wins,"max_loss":round(max_loss,2)})

# Summary table
print(f"\n{'='*70}")
print(f" Stability Ranking")
print(f"{'='*70}")
print(f"{'N':>3}  {'Mean':>7} {'Std':>6} {'Sharpe':>7} {'Wins':>5} {'MaxLoss':>7} {'Stability':>10}")
sdf = pd.DataFrame(summary).sort_values("mean", ascending=False)
for _,r in sdf.iterrows():
    sharpe = r["mean"] / max(r["std"], 0.01)
    score = r["mean"] - 1.5 * r["std"]
    print(f"{r['N']:>3.0f}  {r['mean']:+6.2f}% {r['std']:5.2f}% {sharpe:+7.3f} {r['wins']:>4}/10 {r['max_loss']:+6.2f}% {'★'*max(1,int(score+6))}")
