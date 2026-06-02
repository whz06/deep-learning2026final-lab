"""v8/final_summary.py — v7 Spatial: 扣成本 + 10×10 窗口 + N/K sweep (净值成本).

Cost model: net change — 同股买卖不算费用.
  sell cost: 印花0.05%+佣金0.025%+过户0.001% = 0.076%/支
  buy cost:  佣金0.025%+过户0.001% = 0.026%/支
  cost = (net_sold × 0.076% + net_bought × 0.026%) / N_hold
"""
import numpy as np, pandas as pd, os, sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(SCRIPT_DIR)
SCORES = os.path.join(ROOT, "v7", "results", "daily_scores_spatial_t1.parquet")
PARQUET = os.path.join(ROOT, "processed", "all_data.parquet")
INDEX_P = os.path.join(ROOT, "data", "market", "000300.SH.csv")

SELL_COST = 0.00076  # per stock, as fraction of portfolio (will divide by N)
BUY_COST  = 0.00026
TEST_START, TEST_END = "20260105", "20260529"
CSI5D_THRESH = -1.0; RISK_OFF = 0.80

# Data
csi = pd.read_csv(INDEX_P, dtype={"trade_date": str})
csi_map = dict(zip(csi["trade_date"], pd.to_numeric(csi["pct_chg"], errors="coerce") / 100.0))
csi_dates = sorted(csi_map.keys())
def get_csi5d(d):
    if d not in csi_dates: return 0.0
    idx = csi_dates.index(d)
    return sum(csi_map[csi_dates[i]] for i in range(max(0, idx-4), idx+1)) * 100

df = pd.read_parquet(PARQUET); df["trade_date"] = df["trade_date"].astype(str)
ret_d = {}; mv_d = {}
for d, sdf in df.groupby("trade_date"):
    ret_d[d] = dict(zip(sdf["ts_code"], pd.to_numeric(sdf["pct_chg"], errors="coerce").astype(float)))
    mv_d[d] = dict(zip(sdf["ts_code"], pd.to_numeric(sdf["total_mv"], errors="coerce").astype(float)))

scores = pd.read_parquet(SCORES)
sd = {}
for _, r in scores.iterrows():
    sd.setdefault(r["trade_date"], {})[r["ts_code"]] = r["score"]
dates = [d for d in sorted(sd.keys()) if TEST_START <= d <= TEST_END]
print(f"Period: {dates[0]} ~ {dates[-1]}, {len(dates)} days")

# ===== Backtest with NET CHANGE cost =====
def backtest_net(N, K, pos_fn, label):
    """Backtest with net-change cost: only count stocks that actually leave/enter."""
    holdings = None; daily_p, daily_c, daily_ew = [], [], []
    total_cost, total_sold, total_bought = 0.0, 0, 0

    for si in range(len(dates) - 1):
        fd, rd = dates[si], dates[si+1]
        if fd not in sd or rd not in ret_d: continue
        ss, rr = sd[fd], ret_d[rd]

        sorted_s = sorted(ss.items(), key=lambda x: x[1], reverse=True)
        top = [c for c, _ in sorted_s if c in rr]
        if len(top) < N: continue

        target = max(1, int(round(N * pos_fn(fd))))

        if holdings is None:
            old_set = set()
            holdings = top[:target]
        else:
            old_set = set(holdings)

            # Determine sells: bottom-K from current holdings
            held_s = [(c, ss.get(c, -1e6)) for c in holdings if c in ss]
            held_s.sort(key=lambda x: x[1], reverse=True)
            to_sell = {c for c, _ in held_s[-K:]} if len(held_s) > K else set()

            # Position reduction: sell extra stocks
            extra = len(holdings) - target
            if extra > 0:
                keep = [c for c, _ in held_s if c not in to_sell]
                for c in keep[-extra:]:
                    to_sell.add(c)

            holdings = [c for c in holdings if c not in to_sell]

            # Buy to fill
            held_set = set(holdings)
            for c in top:
                if len(holdings) >= target: break
                if c not in held_set:
                    holdings.append(c)

        # Net change cost: only stocks truly swapped
        new_set = set(holdings)
        net_sold = len(old_set - new_set)
        net_bought = len(new_set - old_set)
        cost = (net_sold * SELL_COST + net_bought * BUY_COST) / max(N, 1)

        total_cost += cost; total_sold += net_sold; total_bought += net_bought

        pr = np.mean([rr.get(c, 0.0) for c in holdings]) if holdings else 0.0
        pr -= cost
        daily_p.append(pr)
        daily_c.append(csi_map.get(rd, 0.0) * 100)

        # Equal-weight benchmark
        ew_rets = list(rr.values())
        daily_ew.append(np.mean(ew_rets) if ew_rets else 0.0)

    if not daily_p: return None
    cum_p = np.sum(daily_p); cum_c = np.sum(daily_c); cum_ew = np.sum(daily_ew)
    arr = np.array(daily_p)
    sharpe = np.mean(arr) / (np.std(arr, ddof=1) + 1e-12) * np.sqrt(252)
    max_dd = np.max(np.maximum.accumulate(np.cumsum(arr)) - np.cumsum(arr))

    # 10×10 windows
    windows_10d = []
    for wi in range(0, len(daily_p) - 9, 10):
        windows_10d.append(np.sum(daily_p[wi:wi+10]))

    return dict(label=label, N=N, K=K, cum=round(cum_p, 2), csi=round(cum_c, 2),
                excess=round(cum_p - cum_c, 2), sharpe=round(sharpe, 3),
                cost=round(total_cost * 100, 2), max_dd=round(max_dd, 2),
                net_sold=int(total_sold), net_bought=int(total_bought),
                days=len(daily_p), windows=windows_10d)


# ===== Run: N=5 K=2 vs K=3, with different thresholds =====
print(f"\n{'='*80}")
print(f" v7 Spatial — 净值成本 (同股买卖不算费) + 10×10 窗口")
print(f"{'='*80}")

results = []
for K in [2, 3]:
    for th_desc, pos_fn in [
        ("th=-1.0% E", lambda d: 0.8 if get_csi5d(d) < -1.0 else 1.0),
        ("th=-0.5% E", lambda d: 0.8 if get_csi5d(d) < -0.5 else 1.0),
        ("Always 100%", lambda d: 1.0),
    ]:
        r = backtest_net(N=5, K=K, pos_fn=pos_fn, label=f"N=5 K={K} {th_desc}")
        if r: results.append(r)

print(f"  {'Strategy':<28} {'Gross':>7} {'Cost':>6} {'Net':>7} {'Excess':>7} {'Sharpe':>6} {'MaxDD':>7}")
print(f"  {'─'*70}")
for r in sorted(results, key=lambda x: -(x['cum'])):
    gross = r['cum'] + r['cost']
    print(f"  {r['label']:<28} {gross:>+6.2f}% {r['cost']:>+5.2f}% {r['cum']:>+6.2f}% "
          f"{r['excess']:>+6.2f}% {r['sharpe']:>6.3f} {r['max_dd']:>+6.2f}%")
    print(f"    net sold={r['net_sold']} bought={r['net_bought']} (real trades)")

# ===== 10×10 windows =====
print(f"\n{'='*80}")
print(f" 10×10 滚动窗口 (每 10 天为一组, net)")
print(f"{'='*80}")

base = [r for r in results if "K=3" in r["label"] and "th=-1.0%" in r["label"]][0]
best = max(results, key=lambda x: x['cum'])

for name, r in [("K=3 th=-1.0% [当前]", base), (f"K=2 th=-1.0% [建议]", best)]:
    ws = r['windows']
    if not ws: continue
    print(f"  {name}:")
    print(f"    {'Win':>4} {'Return':>8} {'Cum':>8}  {'█'*20}")
    cum = 0
    for wi, w in enumerate(ws):
        cum += w
        bars = int(max(0, cum) / 3) if cum > 0 else 0
        print(f"    {wi+1:>4} {w:>+7.2f}% {cum:>+7.2f}%  {'█'*min(bars, 30)}")
    print(f"    均值: {np.mean(ws):+.2f}%  std: {np.std(ws):.2f}%  "
          f"胜率: {np.mean([w>0 for w in ws]):.0%}  "
          f"最差: {min(ws):+.2f}%  最好: {max(ws):+.2f}%")

# ===== Full N/K sweep with net costs =====
print(f"\n{'='*80}")
print(f" N/K Sweep (净值成本, th=-1.0% → 80%)")
print(f"{'='*80}")

sweep = []
for N in [5, 6, 7, 8, 10, 15, 20]:
    for K in [2, 3, 5, 8, 10]:
        if K >= N: continue
        r = backtest_net(N=N, K=K,
                         pos_fn=lambda d: 0.8 if get_csi5d(d) < -1.0 else 1.0,
                         label=f"N={N} K={K}")
        if r: sweep.append(r)

sweep_df = pd.DataFrame(sweep).sort_values("cum", ascending=False)
print(f"  {'N':>3} {'K':>3} {'Gross':>7} {'Cost':>6} {'Net':>7} {'Excess':>7} {'DD':>6} {'Trades':>7}")
for _, r in sweep_df.head(12).iterrows():
    gross = r['cum'] + r['cost']
    print(f"  {r['N']:>3.0f} {r['K']:>3.0f} {gross:>+6.2f}% {r['cost']:>+5.2f}% {r['cum']:>+6.2f}% "
          f"{r['excess']:>+6.2f}% {r['max_dd']:>+5.2f}% {r['net_sold']:>+4}/{r['net_bought']:<4}")

baseline = [r for r in sweep if r['N'] == 5 and r['K'] == 3][0]
best_s = sweep_df.iloc[0]
print(f"\n  当前最优: N={best_s['N']:.0f} K={best_s['K']:.0f} net={best_s['cum']:+.2f}%")
print(f"  vs 当前: delta={best_s['cum'] - baseline['cum']:+.2f}%")
