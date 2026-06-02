"""v8/validate_diversification.py — 行业分散度约束回测

用 v7 Spatial 现有 daily_scores，对比：
  A: 原始选股（按 score 取 top-N）
  B: 行业分散约束（同行业 ≤ 2 只）

回测：Strategy B，Feb-May 2026
"""
import os, sys, glob, numpy as np, pandas as pd

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(SCRIPT_DIR)
SCORES_PATH = os.path.join(ROOT, "v7", "results", "daily_scores_spatial_t1.parquet")
BASIC_PATH = os.path.join(ROOT, "data", "basic.csv")
INDEX_PATH = os.path.join(ROOT, "data", "market", "000300.SH.csv")
PARQUET_PATH = os.path.join(ROOT, "processed", "all_data.parquet")

MAX_PER_INDUSTRY = 2
N_HOLD, K_SELL = 5, 3
CSI5D_THRESH, RISK_OFF = -1.0, 0.80

# ===== Load data =====
print("[div] Loading v7 Spatial scores ...")
scores = pd.read_parquet(SCORES_PATH)
print(f"[div] Scores: {len(scores)} records, {scores['trade_date'].nunique()} dates")

print("[div] Loading industry mapping ...")
basic = pd.read_csv(BASIC_PATH, dtype={"ts_code": str})
basic["industry"] = basic["industry"].fillna("Other")
ts2ind = dict(zip(basic["ts_code"], basic["industry"]))
default_ind = "Other"

# Attach industry to scores
scores["industry"] = scores["ts_code"].map(ts2ind).fillna(default_ind)

# CSI300
csi = pd.read_csv(INDEX_PATH, dtype={"trade_date": str})
csi_map = dict(zip(csi["trade_date"], csi["pct_chg"].astype(float)))
csi_dates = sorted(csi_map.keys())

def get_csi5d(date):
    if date not in csi_dates: return 0.0
    idx = csi_dates.index(date)
    start = max(0, idx - 4)
    return sum(csi_map[csi_dates[i]] for i in range(start, idx + 1))

# Returns
print("[div] Loading returns ...")
df = pd.read_parquet(PARQUET_PATH)
df["trade_date"] = df["trade_date"].astype(str)
ret_d = {}
for d, sdf in df.groupby("trade_date"):
    ret_d[d] = dict(zip(sdf["ts_code"], sdf["pct_chg"].astype(float)))

# ===== Build score dict =====
sd = {}
for _, r in scores.iterrows():
    sd.setdefault(r["trade_date"], {})[r["ts_code"]] = r["score"]

test_dates = sorted(sd.keys())
print(f"[div] Test dates: {test_dates[0]} ~ {test_dates[-1]}, {len(test_dates)} days")

# ===== Backtest engine =====
def backtest(use_diversify, label=""):
    holdings = None
    daily_p, daily_c, daily_n_ind = [], [], []

    for si in range(len(test_dates) - 1):
        fd = test_dates[si]; rd = test_dates[si + 1]
        if fd not in sd or rd not in ret_d: continue
        ss = sd[fd]; rr = ret_d[rd]

        # Sort stocks by score
        sorted_s = sorted(ss.items(), key=lambda x: x[1], reverse=True)

        if use_diversify:
            # Constrained selection: max MAX_PER_INDUSTRY per industry
            selected = []
            ind_count = {}
            for ts_code, sc in sorted_s:
                ind = ts2ind.get(ts_code, default_ind)
                if ind_count.get(ind, 0) >= MAX_PER_INDUSTRY:
                    continue
                if ts_code in rr:
                    selected.append(ts_code)
                    ind_count[ind] = ind_count.get(ind, 0) + 1
                if len(selected) >= 50: break  # keep enough for replacement
            top = selected
        else:
            top = [c for c, _ in sorted_s if c in rr]

        if len(top) < N_HOLD: continue

        csi5d = get_csi5d(fd)
        pos = RISK_OFF if csi5d < CSI5D_THRESH else 1.0
        target = max(1, int(round(N_HOLD * pos)))

        if holdings is None:
            holdings = top[:target]
        else:
            held_s = [(c, ss.get(c, -1e6)) for c in holdings if c in ss]
            held_s.sort(key=lambda x: x[1], reverse=True)
            # Sell worst K
            to_sell = {c for c, _ in held_s[-K_SELL:]} if len(held_s) > K_SELL else set()
            extra = len(holdings) - target
            if extra > 0:
                cand = [c for c, _ in held_s if c not in to_sell]
                for c in cand[-extra:]: to_sell.add(c)
            holdings = [c for c in holdings if c not in to_sell]
            # Buy new from top
            held_set = set(holdings)
            for c in top:
                if len(holdings) >= target: break
                if c not in held_set: holdings.append(c)

        pr = np.mean([rr.get(c, 0.0) for c in holdings]) if holdings else 0.0
        cr = csi_map.get(rd, 0.0)
        daily_p.append(pr); daily_c.append(cr)

        # Track industry concentration
        ind_hold = [ts2ind.get(c, default_ind) for c in holdings]
        ind_counts = pd.Series(ind_hold).value_counts()
        if len(ind_counts) > 0:
            shares = ind_counts / ind_counts.sum()
            hhi = (shares ** 2).sum()
        else:
            hhi = 1.0
        daily_n_ind.append(len(ind_counts))

    if not daily_p: return None

    cum = np.sum(daily_p); cum_c = np.sum(daily_c)
    daily_arr = np.array(daily_p)
    sharpe = np.mean(daily_arr) / (np.std(daily_arr, ddof=1) + 1e-12) * np.sqrt(252)

    # 10-day rolling windows
    windows_10d = []
    for wi in range(0, len(daily_p), 10):
        if wi + 10 <= len(daily_p):
            windows_10d.append(np.sum(daily_p[wi:wi+10]))

    result = {
        "label": label, "cum": round(cum, 4), "excess": round(cum - cum_c, 4),
        "sharpe": round(sharpe, 3), "days": len(daily_p),
        "avg_industries": round(np.mean(daily_n_ind), 2),
        "avg_hhi": round(np.mean(daily_n_ind) / N_HOLD if daily_n_ind else 1.0, 3),
        "worst_window": round(min(windows_10d), 4) if windows_10d else 0,
        "best_window": round(max(windows_10d), 4) if windows_10d else 0,
    }
    return result, daily_p, daily_c

# ===== Run comparison =====
print(f"\n{'='*65}")
print(f" 行业分散度约束 — N={N_HOLD} K={K_SELL} 同行业≤{MAX_PER_INDUSTRY}")
print(f"{'='*65}")

r_orig, dp_orig, dc_orig = backtest(False, "Original")
r_div, dp_div, dc_div = backtest(True, f"Max {MAX_PER_INDUSTRY}/industry")

if r_orig and r_div:
    print(f"\n{'Label':<25} {'Cum':>8} {'Excess':>8} {'Sharpe':>7} {'Avg_Ind':>8} {'Worst10d':>10} {'Best10d':>10}")
    print("-" * 80)
    for r in [r_orig, r_div]:
        print(f"{r['label']:<25} {r['cum']:>+7.2f}% {r['excess']:>+7.2f}% {r['sharpe']:>7.3f} {r['avg_industries']:>8.1f} {r['worst_window']:>+9.2f}% {r['best_window']:>+9.2f}%")

    delta = r_div["cum"] - r_orig["cum"]
    print(f"\n  分散约束增量: {delta:+.2f}% cumulative")
    print(f"  Avg industries held: {r_orig['avg_industries']:.1f} → {r_div['avg_industries']:.1f}")

print(f"\n[div] Done. Summary saved.")
