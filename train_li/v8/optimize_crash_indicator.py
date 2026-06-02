"""v8/optimize_crash_indicator.py — 减仓指标最终优化

E1: CSI5d 阈值扫描 — 全历史周期 (2019-2025)
E5: 全周期回测 — test 期 (Feb-May 2026) 所有候选策略对比

E2/E3/E4 因无历史模型打分数据而跳过（只有 test 期 74 天无法训练阈值）。
"""
import numpy as np, pandas as pd, os

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(SCRIPT_DIR)
PARQUET = os.path.join(ROOT, "processed", "all_data.parquet")
INDEX_P = os.path.join(ROOT, "data", "market", "000300.SH.csv")
SCORES_V7 = os.path.join(ROOT, "v7", "results", "daily_scores_spatial_t1.parquet")

T_TRAIN_START, T_TRAIN_END = "20190102", "20241231"
T_VAL_START, T_VAL_END = "20250102", "20250531"
T_TEST_START, T_TEST_END = "20260105", "20260529"
BETA = 1.1

# ===== Load data =====
csi = pd.read_csv(INDEX_P, dtype={"trade_date": str})
csi["ret"] = pd.to_numeric(csi["pct_chg"], errors="coerce") / 100.0
csi_map = dict(zip(csi["trade_date"], csi["ret"].values * 100))
csi_dates = sorted(csi_map.keys())

def get_csi5d(d):
    if d not in csi_dates: return 0.0
    idx = csi_dates.index(d)
    start = max(0, idx - 4)
    return sum(csi_map[csi_dates[i]] for i in range(start, idx + 1))

# Market features for E4
df = pd.read_parquet(PARQUET)
df["trade_date"] = df["trade_date"].astype(str)

# CSI300 daily return
csi_ret = {d: csi_map.get(d, 0) / 100.0 for d in csi_dates}  # decimal

# Stock returns for full backtest
print("[load] Building return lookup ...")
ret_d = {}
for d, sdf in df.groupby("trade_date"):
    ret_d[d] = dict(zip(sdf["ts_code"],
        pd.to_numeric(sdf["pct_chg"], errors="coerce").astype(float)))

# v7 scores for test
scores_v7 = pd.read_parquet(SCORES_V7)
sd = {}
for _, r in scores_v7.iterrows():
    sd.setdefault(r["trade_date"], {})[r["ts_code"]] = r["score"]
test_dates = sorted(sd.keys())
print(f"[load] Scores: {len(sd)} dates, test={test_dates[0]}~{test_dates[-1]}")


# ===== Backtest engines =====
def backtest_market(dates, pos_fn):
    """Market-level: position × beta × CSI300 return."""
    cum = 0; ro = 0
    for i, d in enumerate(dates[:-1]):
        nx = dates[i + 1]
        if nx not in csi_ret: continue
        pos = pos_fn(d)
        cum += pos * csi_ret[nx] * BETA * 100
        if pos < 1.0: ro += 1
    arr = np.array([pos_fn(dates[i]) * csi_ret.get(dates[i+1], 0) * BETA * 100
                    for i in range(len(dates)-1) if dates[i+1] in csi_ret])
    sharpe = np.mean(arr)/(np.std(arr,ddof=1)+1e-12)*np.sqrt(252)
    return dict(cum=round(cum,3), sharpe=round(sharpe,3), ro_days=ro,
                ro_pct=round(ro/max(len(dates)-1,1)*100,1))


SELL_COST = 0.00076  # 印花税0.05% + 佣金0.025% + 过户费0.001%
BUY_COST  = 0.00026  # 佣金0.025% + 过户费0.001%
N_HOLD, K_SELL = 5, 3

def backtest_stocks(dates, pos_fn):
    """Stock-level: v7 spatial scores, N=5 K=3, position-adjusted, WITH transaction costs."""
    holdings = None; daily_p = []; total_cost = 0.0; debug_n = [0,0]
    for si in range(len(dates)-1):
        fd = dates[si]; rd = dates[si+1]
        if fd not in sd or rd not in ret_d: continue
        ss = sd[fd]; rr = ret_d[rd]
        top = [c for c,_ in sorted(ss.items(),key=lambda x:x[1],reverse=True) if c in rr]
        if len(top) < N_HOLD: continue
        pos = pos_fn(fd)
        target = max(1, int(round(N_HOLD*pos)))
        n_sell, n_buy = 0, 0
        if holdings is None:
            holdings = top[:target]
            n_buy = len(holdings)
        else:
            held_s = [(c, ss.get(c,-1e6)) for c in holdings if c in ss]
            held_s.sort(key=lambda x: x[1], reverse=True)
            to_sell = {c for c,_ in held_s[-K_SELL:]} if len(held_s)>K_SELL else set()
            extra = len(holdings)-target
            if extra>0:
                cand = [c for c,_ in held_s if c not in to_sell]
                for c in cand[-extra:]: to_sell.add(c)
            n_sell = len(to_sell)
            holdings = [c for c in holdings if c not in to_sell]
            held_set = set(holdings)
            for c in top:
                if len(holdings)>=target: break
                if c not in held_set:
                    holdings.append(c)
                    n_buy += 1

        debug_n[0] += n_sell; debug_n[1] += n_buy

        if N_HOLD > 0:
            cost = (n_sell * SELL_COST + n_buy * BUY_COST) / N_HOLD
        else:
            cost = 0.0
        total_cost += cost

        pr = np.mean([rr.get(c,0) for c in holdings]) if holdings else 0.0
        pr -= cost
        daily_p.append(pr)

    if not daily_p: return None
    cum = np.sum(daily_p); arr = np.array(daily_p)
    sharpe = np.mean(arr)/(np.std(arr,ddof=1)+1e-12)*np.sqrt(252)
    # Debug: print trade stats
    print(f"    [trades] sell={debug_n[0]} buy={debug_n[1]} days={len(daily_p)} cost={total_cost*100:.2f}%")
    return dict(cum=round(cum,3), sharpe=round(sharpe,3), days=len(daily_p),
                total_cost=round(total_cost*100,3))


# ============================================================
# E1: CSI5d threshold sweep — full history
# ============================================================
print(f"\n{'='*65}")
print(" E1: CSI5d threshold sweep (2019-2025 full history)")
print(f"{'='*65}")

all_dates = [d for d in csi_dates if T_TRAIN_START <= d <= T_VAL_END]
train_dates = [d for d in csi_dates if T_TRAIN_START <= d <= T_TRAIN_END]
val_dates = [d for d in csi_dates if T_VAL_START <= d <= T_VAL_END]

print(f"  Train: {train_dates[0]}~{train_dates[-1]}, {len(train_dates)} days")
print(f"  Val:   {val_dates[0]}~{val_dates[-1]}, {len(val_dates)} days")

# Sweep thresholds on full all_dates
results_e1 = []
for th in np.arange(-6.0, 0.51, 0.2):
    pf = lambda d, t=th: 0.8 if get_csi5d(d) < t else 1.0
    r = backtest_market(all_dates, pf)
    r["threshold"] = round(th, 1)
    results_e1.append(r)

# Find best overall
base_all = [r for r in results_e1 if r["threshold"] == -1.0][0]
best_all = max(results_e1, key=lambda x: x["cum"])

print(f"\n  Full period (2019-2025):")
print(f"  Baseline th=-1.0:  cum={base_all['cum']:+.1f}% sharpe={base_all['sharpe']} ro_days={base_all['ro_days']}")
print(f"  Best    th={best_all['threshold']:.1f}: cum={best_all['cum']:+.1f}% sharpe={best_all['sharpe']} ro_days={best_all['ro_days']}")
print(f"  Delta: {best_all['cum']-base_all['cum']:+.1f}%")

# Also show for each year separately
for label, dates_sub in [("Train(19-24)", train_dates), ("Val(2025)", val_dates)]:
    r_base = backtest_market(dates_sub, lambda d: 0.8 if get_csi5d(d) < -1.0 else 1.0)
    r_never = backtest_market(dates_sub, lambda d: 1.0)
    print(f"\n  {label}:")
    print(f"    Always 100%:  cum={r_never['cum']:+.1f}%")
    print(f"    th=-1.0:      cum={r_base['cum']:+.1f}% (delta={r_base['cum']-r_never['cum']:+.1f}%)")

    # Best for this sub-period
    sub_res = []
    for th in np.arange(-6.0, 0.51, 0.2):
        pf = lambda d, t=th: 0.8 if get_csi5d(d) < t else 1.0
        r = backtest_market(dates_sub, pf); r["th"] = round(th,1); sub_res.append(r)
    best_sub = max(sub_res, key=lambda x: x["cum"])
    print(f"    Best th={best_sub['th']:.1f}: cum={best_sub['cum']:+.1f}% (delta vs always 100%: {best_sub['cum']-r_never['cum']:+.1f}%)")
    print(f"    Top thresholds: " + " ".join(
        f"th={r['th']:+.1f}({r['cum']:+.1f})" for r in sorted(sub_res, key=lambda x:-x["cum"])[:5]))

# Asymmetry analysis
print(f"\n  Asymmetry analysis (why th=-1.0 works):")
# On days where csi5d < -1.0, what's the NEXT day's CSI300 return?
crash_days = [d for d in all_dates if get_csi5d(d) < -1.0]
crash_next_ret = []
for d in crash_days:
    idx = csi_dates.index(d)
    if idx + 1 < len(csi_dates):
        crash_next_ret.append(csi_ret[csi_dates[idx + 1]])
if crash_next_ret:
    arr = np.array(crash_next_ret) * 100
    print(f"  After csi5d < -1%: mean next-day ret = {arr.mean():+.2f}% std={arr.std():.2f}%")
    print(f"  days={len(arr)} pos_days={(arr>0).mean():.0%}")
    # Benefit of reducing: saved 20% × loss on down days
    down = arr[arr < 0]
    up = arr[arr > 0]
    saved = -0.2 * down.sum() if len(down) > 0 else 0
    lost = 0.2 * up.sum() if len(up) > 0 else 0
    print(f"  Saved from down days: {saved:+.2f}% | Lost on up days: {lost:+.2f}%")
    print(f"  Net benefit: {saved-lost:+.2f}%")


# ============================================================
# E5: Full backtest — all strategies on test
# ============================================================
print(f"\n{'='*65}")
print(" E5: Full stock-level backtest (Feb-May 2026)")
print(f"{'='*65}")

strategies = [
    ("Always 100%", lambda d: 1.0),
    ("Always 80%", lambda d: 0.8),
    ("th=-0.5% → 80%", lambda d: 0.8 if get_csi5d(d) < -0.5 else 1.0),
    ("th=-1.0% → 80% [当前]", lambda d: 0.8 if get_csi5d(d) < -1.0 else 1.0),
    ("th=-1.5% → 80%", lambda d: 0.8 if get_csi5d(d) < -1.5 else 1.0),
    ("th=-2.0% → 80%", lambda d: 0.8 if get_csi5d(d) < -2.0 else 1.0),
    ("th=-3.0% → 80%", lambda d: 0.8 if get_csi5d(d) < -3.0 else 1.0),
    # Adaptive: tighter thresholds = more reduction
    ("Adapt: -0.5→60%/-1.5→80%/-3→80%", lambda d: (
        0.6 if get_csi5d(d) < -0.5 else
        0.8 if get_csi5d(d) < -1.5 else
        1.0)),
]

print(f"  {'Strategy':<30} {'Cum':>8} {'Sharpe':>7} {'RiskOff%':>8}")
print("  " + "-" * 55)
all_r = []
for name, pf in strategies:
    r = backtest_stocks(test_dates, pf)
    if r:
        # Get risk-off %
        ro = sum(1 for d in test_dates if pf(d) < 1.0)
        r["name"] = name; r["ro_pct"] = round(ro/len(test_dates)*100, 1)
        all_r.append(r)
        marker = " <-- current" if "当前" in name else ""
        print(f"  {name:<30} {r['cum']:>+7.2f}% {r['sharpe']:>7.3f} {r['ro_pct']:>7.1f}% {r.get('total_cost',0):>+6.2f}%")

base = [r for r in all_r if "当前" in r["name"]][0]
best = max(all_r, key=lambda x: x["cum"])
print(f"\n  Baseline: {base['cum']:+.2f}%")
print(f"  Best:     {best['cum']:+.2f}% ({best['name']})")
print(f"  Delta:    {best['cum']-base['cum']:+.2f}%")
