"""v9/phase0_feasibility.py — Test if regime-conditional vol improves sell decisions.

Test: sell_score = -norm(v7_score) + g(regime) * λ * norm(vol_20d)
      g(bull)=-1, g(neutral)=0, g(bear)=+1  (variant A)
      bear = +norm(σ) only (variant B, pure vol)
"""
import numpy as np, pandas as pd, os, sys, time

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(SCRIPT_DIR)
SCORES = os.path.join(ROOT, "v7", "results", "daily_scores_spatial_t1.parquet")
PARQUET = os.path.join(ROOT, "processed", "all_data.parquet")
INDEX_P = os.path.join(ROOT, "data", "market", "000300.SH.csv")

SELL_COST, BUY_COST = 0.00076, 0.00026
CSI5D_THRESH = -1.0; RISK_OFF = 0.80
N, K = 5, 3

t0 = time.time()

# ── CSI300 ──
csi = pd.read_csv(INDEX_P, dtype={"trade_date": str})
csi_map = dict(zip(csi["trade_date"], pd.to_numeric(csi["pct_chg"], errors="coerce") / 100.0))
csi_dates = sorted(csi_map.keys())

def get_csi5d(d):
    if d not in csi_dates: return 0.0
    idx = csi_dates.index(d)
    return sum(csi_map[csi_dates[i]] for i in range(max(0, idx - 4), idx + 1)) * 100

# ── v7 scores ──
scores = pd.read_parquet(SCORES)
sd = {}
for _, r in scores.iterrows():
    sd.setdefault(r["trade_date"], {})[r["ts_code"]] = r["score"]

test_dates = sorted([d for d in sd if "2026" in d])
print(f"[{time.time()-t0:.0f}s] v7 scores loaded: {len(test_dates)} test dates ({test_dates[0]} ~ {test_dates[-1]})")

# ── Load only needed date range from parquet ──
df = pd.read_parquet(PARQUET, columns=["trade_date", "ts_code", "pct_chg"])
df["trade_date"] = df["trade_date"].astype(str)
# Filter to dates >= 20251101 for vol_20d lookback
df = df[df["trade_date"] >= "20251101"]
print(f"[{time.time()-t0:.0f}s] Parquet loaded: {len(df)} rows, dates {df.trade_date.min()}~{df.trade_date.max()}")

# ── Returns ──
ret_d = {}
for d, sdf in df.groupby("trade_date"):
    ret_d[d] = dict(zip(sdf["ts_code"], pd.to_numeric(sdf["pct_chg"], errors="coerce").astype(float)))

all_dates = sorted(ret_d.keys())
print(f"[{time.time()-t0:.0f}s] ret_d built: {len(all_dates)} dates ({all_dates[0]} ~ {all_dates[-1]})")

# ── vol_20d: rolling 20-day std of daily returns (backward-looking) ──
vol_d = {}
for i, d in enumerate(all_dates):
    start = max(0, i - 19)  # 20 days including today? No, last 20 up to d-1
    lookback = all_dates[start:i + 1]  # include today for convenience, won't matter much
    if len(lookback) < 5:
        vol_d[d] = {}
        continue
    # Collect all codes with returns
    all_codes = set()
    for lb in lookback:
        all_codes.update(ret_d[lb].keys())
    vd = {}
    for c in all_codes:
        r_vals = []
        for lb in lookback:
            rv = ret_d[lb].get(c, None)
            if rv is not None and not np.isnan(rv):
                r_vals.append(rv)
        if len(r_vals) >= 5:
            vd[c] = float(np.std(r_vals, ddof=1))
    vol_d[d] = vd

print(f"[{time.time()-t0:.0f}s] vol_d built: {len(vol_d)} dates")

# ── Filter to test period ──
dates = [d for d in test_dates if d in ret_d and d in vol_d and vol_d[d]]
print(f"[{time.time()-t0:.0f}s] Ready: {len(dates)} test days ready for backtest")

# ── Backtest engine ──
def backtest_v9(sell_score_fn, label):
    holdings = None
    daily_p, daily_c, daily_ew = [], [], []
    total_cost, total_sold, total_bought = 0.0, 0, 0

    for si in range(len(dates) - 1):
        fd, rd = dates[si], dates[si + 1]
        if fd not in sd or rd not in ret_d: continue
        ss, rr, vv = sd[fd], ret_d[rd], vol_d.get(fd, {})
        csi5d = get_csi5d(fd)

        sorted_s = sorted(ss.items(), key=lambda x: x[1], reverse=True)
        top = [c for c, _ in sorted_s if c in rr and c in vv]
        if len(top) < N: continue

        target = max(1, int(round(N * (RISK_OFF if csi5d < CSI5D_THRESH else 1.0))))

        if holdings is None:
            old_set = set()
            holdings = top[:target]
        else:
            old_set = set(holdings)

            # Sell decision
            held_info = []
            for c in holdings:
                if c in ss and c in vv:
                    held_info.append((c, ss[c], vv[c], csi5d))

            if len(held_info) >= K:
                r_arr = np.array([x[1] for x in held_info])
                v_arr = np.array([x[2] for x in held_info])
                csi_arr = np.array([x[3] for x in held_info])

                r_norm = (r_arr - r_arr.mean()) / (r_arr.std() + 1e-8)
                v_norm = (v_arr - v_arr.mean()) / (v_arr.std() + 1e-8)

                s_scores = sell_score_fn(r_norm, v_norm, csi_arr)
                order = np.argsort(s_scores)[::-1]  # higher = sell
                to_sell = {held_info[i][0] for i in order[:K]}
            else:
                to_sell = set()

            # Position reduction
            extra = len(holdings) - target
            if extra > 0 and len(held_info) >= K:
                r_arr = np.array([x[1] for x in held_info])
                v_arr = np.array([x[2] for x in held_info])
                csi_arr = np.array([x[3] for x in held_info])
                r_norm = (r_arr - r_arr.mean()) / (r_arr.std() + 1e-8)
                v_norm = (v_arr - v_arr.mean()) / (v_arr.std() + 1e-8)
                s_scores = sell_score_fn(r_norm, v_norm, csi_arr)
                order = np.argsort(s_scores)[::-1]
                for i in order:
                    c = held_info[i][0]
                    if c not in to_sell:
                        to_sell.add(c)
                        extra -= 1
                        if extra <= 0: break

            holdings = [c for c in holdings if c not in to_sell]
            held_set = set(holdings)
            for c in top:
                if len(holdings) >= target: break
                if c not in held_set:
                    holdings.append(c)

        new_set = set(holdings)
        net_sold = len(old_set - new_set)
        net_bought = len(new_set - old_set)
        cost = (net_sold * SELL_COST + net_bought * BUY_COST) / max(N, 1)
        total_cost += cost; total_sold += net_sold; total_bought += net_bought

        pr = np.mean([rr.get(c, 0.0) for c in holdings]) if holdings else 0.0
        pr -= cost
        daily_p.append(pr)
        daily_c.append(csi_map.get(rd, 0.0) * 100)
        daily_ew.append(np.mean(list(rr.values())) if rr else 0.0)

    if not daily_p: return None
    cum_p = np.sum(daily_p); cum_c = np.sum(daily_c)
    arr = np.array(daily_p)
    sharpe = np.mean(arr) / (np.std(arr, ddof=1) + 1e-12) * np.sqrt(252)
    max_dd = np.max(np.maximum.accumulate(np.cumsum(arr)) - np.cumsum(arr))
    windows_10d = [np.sum(daily_p[wi:wi+10]) for wi in range(0, len(daily_p)-9, 10)]

    return dict(label=label, cum=round(cum_p, 2), gross=round(cum_p+total_cost*100, 2),
                csi=round(cum_c, 2), excess=round(cum_p-cum_c, 2),
                sharpe=round(sharpe, 3), cost=round(total_cost*100, 2),
                max_dd=round(max_dd, 2), net_sold=int(total_sold),
                net_bought=int(total_bought), days=len(daily_p), windows=windows_10d)


# ═══════════════════════════════════════════════════════════════
# RUN EXPERIMENTS
# ═══════════════════════════════════════════════════════════════

results = []

# Baseline: sell by v7 score
def fn_baseline(r_norm, v_norm, csi_arr):
    return -r_norm
results.append(backtest_v9(fn_baseline, "baseline (v7 score only)"))

# Variant A: g(bull)=-1, g(neutral)=0, g(bear)=+1
for lam in [0.0, 0.5, 1.0, 2.0, 3.0, 5.0, 10.0]:
    def make_a(l):
        def fn(r_norm, v_norm, csi_arr):
            g = np.where(csi_arr > 0, -1, np.where(csi_arr < -1.0, 1, 0))
            return -r_norm + g * l * v_norm
        return fn
    results.append(backtest_v9(make_a(lam), f"λ={lam} (r+g*vol)"))

# Variant B: bear = pure vol
for lam in [0.5, 1.0, 2.0, 3.0, 5.0]:
    def make_b(l):
        def fn(r_norm, v_norm, csi_arr):
            s = np.zeros_like(r_norm)
            bull = csi_arr > 0
            neutral = (csi_arr >= -1.0) & (csi_arr <= 0)
            bear = csi_arr < -1.0
            s[bull] = -r_norm[bull] - l * v_norm[bull]
            s[neutral] = -r_norm[neutral]
            s[bear] = v_norm[bear]
            return s
        return fn
    results.append(backtest_v9(make_b(lam), f"λ={lam} (bear_pure_vol)"))

results = [r for r in results if r]

# ═══════════════════════════════════════════════════════════════
# PRINT
# ═══════════════════════════════════════════════════════════════
print(f"\n{'='*85}")
print(f" Phase 0: v7 N={N} K={K} + regime-conditional vol sell ({len(dates)} test days)")
print(f"{'='*85}\n")
print(f"  {'Strategy':<30} {'Gross':>7} {'Cost':>6} {'Net':>7} {'Excess':>7} {'Sharpe':>6} {'MaxDD':>7}")
print(f"  {'─'*72}")

for r in sorted(results, key=lambda x: -(x['cum'])):
    g = r.get('gross', r['cum'] + r['cost'])
    print(f"  {r['label']:<30} {g:>+6.2f}% {r['cost']:>+5.2f}% {r['cum']:>+6.2f}% "
          f"{r['excess']:>+6.2f}% {r['sharpe']:>6.3f} {r['max_dd']:>+6.2f}% "
          f"({r['net_sold']}s/{r['net_bought']}b)")

# ── 10×10 windows ──
print(f"\n{'='*85}")
print(f" 10×10 Rolling Windows (net %)")
print(f"{'='*85}")

baseline_r = [r for r in results if "baseline" in r["label"]][0]
best = max(results, key=lambda x: x['cum'])

for name, r in [("Baseline (v7 score)", baseline_r), (f"Best: {best['label']}", best)]:
    ws = r['windows']
    if not ws: continue
    print(f"\n  {name}:")
    cum = 0
    for wi, w in enumerate(ws):
        cum += w
        print(f"    {wi+1:>4} {w:>+7.2f}% {cum:>+7.2f}%")
    print(f"    均值: {np.mean(ws):+.2f}%  σ: {np.std(ws):.2f}%  "
          f"胜率: {np.mean([w>0 for w in ws]):.0%}")

# ── Summary ──
print(f"\n{'='*85}")
print(f" DELTA vs BASELINE")
print(f"{'='*85}")
base = baseline_r['cum']
for r in sorted(results, key=lambda x: -(x['cum'])):
    d = r['cum'] - base
    print(f"  {r['label']:<30}  {r['cum']:>+6.2f}%  (Δ = {d:>+6.2f}%)")
print(f"\n  Total time: {time.time()-t0:.0f}s")
