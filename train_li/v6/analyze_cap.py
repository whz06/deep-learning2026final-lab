"""1. Analyze market cap of v6 top picks vs all stocks.
2. Run strategy sweep on large-cap filtered universe.
"""
import pandas as pd, numpy as np

SCORES_PATH = r"D:\Workspace\DL_HW\deep-learning2026final-lab\train_li\v6\results\daily_scores.parquet"
PARQUET_PATH = r"D:\Workspace\DL_HW\deep-learning2026final-lab\train_li\processed\all_data.parquet"
INDEX_PATH = r"D:\Workspace\DL_HW\deep-learning2026final-lab\train_li\data\market\000300.SH.csv"

scores = pd.read_parquet(SCORES_PATH)
dates_s = sorted(scores["trade_date"].unique())

all_needed = set(dates_s)
for d in dates_s[:-1]:
    all_needed.add(dates_s[dates_s.index(d)+1])

df = pd.read_parquet(PARQUET_PATH, filters=[("trade_date", "in", sorted(all_needed))])
df["trade_date"] = df["trade_date"].astype(str)

csi = pd.read_csv(INDEX_PATH, dtype={"trade_date": str})
csi_map = dict(zip(csi["trade_date"], csi["pct_chg"].astype(float)))
csi_dates = sorted(csi_map.keys())

def get_csi5d(date):
    if date not in csi_dates: return 0.0
    idx = csi_dates.index(date)
    return sum(csi_map[csi_dates[i]] for i in range(max(0,idx-4), idx+1))

# =========== Part 1: Market cap analysis ===========
print("=" * 60)
print(" Part 1: Market Cap Analysis")
print("=" * 60)

# Build daily total_mv lookup
mv_by_date = {}
for _, row in df.iterrows():
    mv_by_date.setdefault(row["trade_date"], {})[row["ts_code"]] = float(row.get("total_mv", -1))

score_by_date = {}
for _, row in scores.iterrows():
    score_by_date.setdefault(row["trade_date"], {})[row["ts_code"]] = row["score"]

mv_pctiles = []  # percentile of top-20 within the market cap distribution

for i, d in enumerate(dates_s[:-1]):
    if d not in score_by_date or d not in mv_by_date:
        continue
    day_scores = score_by_date[d]
    day_mv = mv_by_date[d]
    
    # Skip stocks without MV data
    valid = {c: s for c, s in day_scores.items() if c in day_mv and day_mv[c] > 0}
    if len(valid) < 100:
        continue
    
    sorted_by_score = sorted(valid.items(), key=lambda x: x[1], reverse=True)
    top20 = [c for c, _ in sorted_by_score[:20]]
    
    # Rank stocks by market cap (descending)
    all_mv = [(c, day_mv[c]) for c in valid]
    all_mv.sort(key=lambda x: x[1], reverse=True)
    mv_ranks = {c: rank/len(all_mv) for rank, (c, _) in enumerate(all_mv)}
    
    # What percentile are top-20 picks in MV distribution?
    for c in top20:
        if c in mv_ranks:
            mv_pctiles.append(mv_ranks[c])

mv_pctiles = np.array(mv_pctiles)
print(f"  v6 top-20 market cap percentiles (0=smallest, 1=largest):")
print(f"    Mean:   {mv_pctiles.mean():.3f}")
print(f"    Median: {np.median(mv_pctiles):.3f}")
print(f"    P25:    {np.percentile(mv_pctiles, 25):.3f}")
print(f"    P75:    {np.percentile(mv_pctiles, 75):.3f}")
print(f"    % Top 20% MV: {(mv_pctiles > 0.8).mean()*100:.1f}%")
print(f"    % Top 50% MV: {(mv_pctiles > 0.5).mean()*100:.1f}%")
print(f"    % Bottom 20% MV: {(mv_pctiles < 0.2).mean()*100:.1f}%")
print(f"    % Bottom 50% MV: {(mv_pctiles < 0.5).mean()*100:.1f}%")

# =========== Part 2: Strategy sweep on large-cap only ===========
print(f"\n{'='*60}")
print(" Part 2: Large-Cap Filtered Strategy Sweep")
print(f"{'='*60}")

# Filter: keep only top 20% by market cap on each date
ret_by_date = {}
for _, row in df.iterrows():
    ret_by_date.setdefault(row["trade_date"], {})[row["ts_code"]] = float(row["pct_chg"])

def run_sweep(mv_filter_percentile=0.8):
    """Run sweep keeping only stocks above given MV percentile."""
    results = []
    N_HOLDS = [5, 10, 15, 20, 25, 30]
    SELL_KS = [2, 3, 5, 8, 10]
    
    for N in N_HOLDS:
        for K in SELL_KS:
            if K >= N: continue
            
            holdings = []
            daily_p, daily_c, daily_e = [], [], []
            pos_hist = []
            
            for i in range(len(dates_s) - 1):
                fd, rd = dates_s[i], dates_s[i+1]
                if fd not in score_by_date or fd not in mv_by_date or rd not in ret_by_date:
                    continue
                
                day_scores = score_by_date[fd]
                day_mv = mv_by_date[fd]
                day_rets = ret_by_date[rd]
                
                # Filter to large-cap only
                stocks_mv = [(c, day_mv[c]) for c in day_scores if c in day_mv and day_mv[c] > 0 and c in day_rets]
                if len(stocks_mv) < N:
                    continue
                stocks_mv.sort(key=lambda x: x[1], reverse=True)
                cutoff_idx = int(len(stocks_mv) * mv_filter_percentile)
                if cutoff_idx >= len(stocks_mv): cutoff_idx = len(stocks_mv) - 1
                mv_cutoff = stocks_mv[cutoff_idx][1]
                
                large_codes = [c for c, mv in stocks_mv if mv >= mv_cutoff]
                if len(large_codes) < N:
                    continue
                
                # Score-filtered
                sc = [(c, day_scores[c]) for c in large_codes]
                sc.sort(key=lambda x: x[1], reverse=True)
                top = [c for c, _ in sc]
                
                csi5d = get_csi5d(fd)
                pos = 0.80 if csi5d < -1.0 else 1.0
                target = max(1, int(round(N * pos)))
                
                if not holdings:
                    holdings = top[:target]
                else:
                    held_s = [(c, day_scores.get(c, -1e6)) for c in holdings if c in day_scores]
                    held_s.sort(key=lambda x: x[1], reverse=True)
                    to_sell = {c for c, _ in held_s[-K:]} if len(held_s) > K else set()
                    extra = len(holdings) - target
                    if extra > 0:
                        cand = [c for c, _ in held_s if c not in to_sell]
                        for c in cand[-extra:]: to_sell.add(c)
                    holdings = [c for c in holdings if c not in to_sell]
                    held_set = set(holdings)
                    for c in top:
                        if len(holdings) >= target: break
                        if c not in held_set: holdings.append(c)
                
                pos_hist.append(pos)
                pr = np.mean([day_rets.get(c, 0.0) for c in holdings]) if holdings else 0.0
                cr = csi_map.get(rd, 0.0)
                er = np.mean(list(day_rets.values())) if day_rets else 0.0
                daily_p.append(pr)
                daily_c.append(cr)
                daily_e.append(er)
            
            if not daily_p: continue
            
            pr_a = np.array(daily_p)
            cr_a = np.array(daily_c)
            er_a = np.array(daily_e)
            cum_p = float(np.sum(pr_a))
            cum_c = float(np.sum(cr_a))
            cum_e = float(np.sum(er_a))
            ex_csi = cum_p - cum_c
            ex_ew = cum_p - cum_e
            ex_d = pr_a - cr_a
            sharpe = float(np.mean(ex_d) / (np.std(ex_d) + 1e-8) * np.sqrt(252))
            cum_series = np.cumsum(pr_a)
            max_dd = float(np.max(np.maximum.accumulate(cum_series) - cum_series))
            
            results.append({
                "mv_filter": mv_filter_percentile,
                "N": N, "K": K,
                "cum": round(cum_p, 4), "csi_cum": round(cum_c, 4), "ew_cum": round(cum_e, 4),
                "ex_csi": round(ex_csi, 4), "ex_ew": round(ex_ew, 4),
                "sharpe": round(sharpe, 4), "max_dd": round(max_dd, 4),
                "days": len(daily_p), "avg_pos": round(np.mean(pos_hist), 3),
            })
    return results

# Run sweep on large-cap (top 20%)
results = run_sweep(0.8)

# Best by excess vs EW
df_res = pd.DataFrame(results).sort_values("ex_ew", ascending=False)

print(f"\n  Large-cap (top 20% MV) sweep results:")
print(f"{'N':>4} {'K':>4} {'Cum':>8} {'ExEW':>8} {'ExCSI':>8} {'Sharpe':>7} {'DD':>6} {'Days':>5}")
print("-" * 60)
for _, r in df_res.head(10).iterrows():
    print(f"{r['N']:>4.0f} {r['K']:>4.0f} {r['cum']:>+7.2f}% {r['ex_ew']:>+7.2f}% "
          f"{r['ex_csi']:>+7.2f}% {r['sharpe']:>7.3f} {r['max_dd']:>5.1f}% {r['days']:>5}")

best = df_res.iloc[0]
print(f"\n  BEST: N={best['N']}, K={best['K']}: cum={best['cum']:+.2f}% ex_ew={best['ex_ew']:+.2f}%")

# Save
OUT = r"D:\Workspace\DL_HW\deep-learning2026final-lab\train_li\v6\results\sweep_largecap.csv"
df_res.to_csv(OUT, index=False)
print(f"\n  Saved -> {OUT}")
