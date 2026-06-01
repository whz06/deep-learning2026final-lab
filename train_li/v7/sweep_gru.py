"""Quick sweep on v7 GRU scores."""
import pandas as pd, numpy as np

SD = r"D:\Workspace\DL_HW\deep-learning2026final-lab\train_li\v7\results\daily_scores_gru_t1.parquet"
PARQUET = r"D:\Workspace\DL_HW\deep-learning2026final-lab\train_li\processed\all_data.parquet"
INDEX_P = r"D:\Workspace\DL_HW\deep-learning2026final-lab\train_li\data\market\000300.SH.csv"

scores = pd.read_parquet(SD)
dates_s = sorted(scores["trade_date"].unique())
needed = set(dates_s)
for i,d in enumerate(dates_s):
    if i+1 < len(dates_s): needed.add(dates_s[i+1])
df = pd.read_parquet(PARQUET); df["trade_date"] = df["trade_date"].astype(str)
df = df[df["trade_date"].isin(needed)]
ret_d = {}
for d in sorted(needed):
    sub = df[df["trade_date"]==d]
    ret_d[d] = dict(zip(sub["ts_code"], sub["pct_chg"].astype(float)))

csi = pd.read_csv(INDEX_P, dtype={"trade_date":str})
csi_map = dict(zip(csi["trade_date"], csi["pct_chg"].astype(float)))
csi_dates = sorted(csi_map.keys())
def get_csi5d(date):
    if date not in csi_dates: return 0.0
    idx=csi_dates.index(date)
    return sum(csi_map[csi_dates[i]] for i in range(max(0,idx-4),idx+1))

sd = {}
for _, row in scores.iterrows():
    sd.setdefault(row["trade_date"], {})[row["ts_code"]] = row["score"]
test_list = sorted(sd.keys())

results = []
for N in [5,10,15,20,25,30]:
    for K in [2,3,5,8,10]:
        if K >= N: continue
        holdings = None; daily_p, daily_c = [], []
        for si in range(len(test_list)-1):
            fd, rd = test_list[si], test_list[si+1]
            if fd not in sd or rd not in ret_d: continue
            ss = sd[fd]; rr = ret_d[rd]
            sorted_s = sorted(ss.items(), key=lambda x:x[1], reverse=True)
            top = [c for c,_ in sorted_s if c in rr]
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
            cr = csi_map.get(rd, 0.0)
            daily_p.append(pr); daily_c.append(cr)
        if daily_p:
            cum = np.sum(daily_p); cum_c = np.sum(daily_c)
            results.append({"N":N,"K":K,"cum":round(cum,4),"excess":round(cum-cum_c,4)})

res = pd.DataFrame(results).sort_values("cum", ascending=False)
print(f"{'N':>4} {'K':>4} {'Cum':>8} {'Excess':>8}")
for _, r in res.head(10).iterrows():
    print(f"{r['N']:>4.0f} {r['K']:>4.0f} {r['cum']:>+7.2f}% {r['excess']:>+7.2f}%")
print(f"\n  BEST: N={res.iloc[0]['N']}, K={res.iloc[0]['K']}: cum={res.iloc[0]['cum']:+.2f}%")
