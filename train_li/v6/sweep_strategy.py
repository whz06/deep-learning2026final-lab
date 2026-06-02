"""v6/sweep_strategy.py — Sweep (N_hold, sell_k) combos with Strategy B.

Uses filtered parquet loading for speed.
Tests holding top-N with daily rotation of bottom-K.
Strategy B: csi5d < -1.0% -> 80% position.
"""
import os, sys, numpy as np, pandas as pd

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(SCRIPT_DIR)

SCORES_PATH = os.path.join(SCRIPT_DIR, "results", "daily_scores.parquet")
PARQUET_PATH = os.path.join(ROOT, "processed", "all_data.parquet")
INDEX_PATH   = os.path.join(ROOT, "data", "market", "000300.SH.csv")
OUT_PATH = os.path.join(SCRIPT_DIR, "results", "strategy_sweep.csv")

# Strategy B
CSI5D_THRESHOLD = -1.0
RISK_OFF_POS = 0.80

# Grid
N_HOLDS = [5, 10, 15, 20, 25, 30]
SELL_KS = [2, 3, 5, 8, 10]

print("[sweep] Loading scores ...")
scores = pd.read_parquet(SCORES_PATH)
dates_s = sorted(scores["trade_date"].unique())
print(f"  {len(dates_s)} score dates, {len(scores):,} rows")

# --- Load CSI300 ---
csi = pd.read_csv(INDEX_PATH, dtype={"trade_date": str})
csi_map = dict(zip(csi["trade_date"], csi["pct_chg"].astype(float)))
csi_dates = sorted(csi_map.keys())

def get_csi5d(date):
    if date not in csi_dates:
        return 0.0
    idx = csi_dates.index(date)
    start = max(0, idx - 4)
    vals = [csi_map[csi_dates[i]] for i in range(start, idx + 1)]
    return sum(v for v in vals if not np.isnan(v))

# --- Load returns (filtered to needed dates) ---
needed_dates = set(dates_s)
for d in dates_s[:-1]:
    needed_dates.add(dates_s[dates_s.index(d) + 1])
print(f"[sweep] Loading returns for {len(needed_dates)} dates ...")
df = pd.read_parquet(PARQUET_PATH, filters=[("trade_date", "in", sorted(needed_dates))])
df["trade_date"] = df["trade_date"].astype(str)

# Build fast return + group lookups
ret_by_date = {}
groups = {}
for ts_code, sdf in df.groupby("ts_code"):
    sdf = sdf.sort_values("trade_date")
    groups[ts_code] = sdf
    for _, row in sdf.iterrows():
        ret_by_date.setdefault(row["trade_date"], {})[ts_code] = float(row["pct_chg"])
del df
print(f"  {len(groups)} stocks, {len(ret_by_date)} return dates")

# Build score dict
score_by_date = {}
for _, row in scores.iterrows():
    score_by_date.setdefault(row["trade_date"], {})[row["ts_code"]] = row["score"]

# --- Backtest ---
def backtest(N_hold, sell_k, verbose=False):
    holdings = []
    daily_port, daily_csi, daily_ew = [], [], []
    pos_history = []
    trades = 0
    valid_days = 0

    for i in range(len(dates_s) - 1):
        feat_date = dates_s[i]
        ret_date = dates_s[i + 1]

        if feat_date not in score_by_date or ret_date not in ret_by_date:
            continue
        day_scores = score_by_date[feat_date]
        day_rets = ret_by_date[ret_date]

        # CSI5d
        csi5d = get_csi5d(feat_date)
        position = RISK_OFF_POS if csi5d < CSI5D_THRESHOLD else 1.0
        target_n = max(1, int(round(N_hold * position)))

        sorted_all = sorted(day_scores.items(), key=lambda x: x[1], reverse=True)
        top_codes = [c for c, _ in sorted_all if c in day_rets]
        if len(top_codes) < N_hold:
            continue

        # Init portfolio
        if not holdings:
            holdings = top_codes[:target_n].copy()
        else:
            # Score current holdings
            held_scores = [(c, day_scores.get(c, -1e6)) for c in holdings if c in day_scores]
            held_scores.sort(key=lambda x: x[1], reverse=True)

            # Sell bottom-K
            to_sell = set()
            if len(held_scores) > sell_k:
                to_sell = {c for c, _ in held_scores[-sell_k:]}

            # Position reduction: sell extra if needed
            need_remove = len(holdings) - target_n
            if need_remove > 0:
                candidates = [c for c, _ in held_scores if c not in to_sell]
                for c in candidates[-need_remove:]:
                    to_sell.add(c)

            if to_sell:
                trades += 1
            holdings = [c for c in holdings if c not in to_sell]

            # Buy to fill
            held_set = set(holdings)
            for c in top_codes:
                if len(holdings) >= target_n:
                    break
                if c not in held_set:
                    holdings.append(c)

        valid_days += 1
        pos_history.append(position)

        # Returns
        held_rets = [day_rets.get(c, 0.0) for c in holdings]
        port_ret = np.mean(held_rets) if held_rets else 0.0
        csi_ret = csi_map.get(ret_date, 0.0)
        ew_ret = np.mean(list(day_rets.values())) if day_rets else 0.0

        daily_port.append(port_ret)
        daily_csi.append(csi_ret)
        daily_ew.append(ew_ret)

    if not daily_port:
        return None

    pr = np.array(daily_port)
    cr = np.array(daily_csi)
    er = np.array(daily_ew)

    cum_p = float(np.sum(pr))
    cum_c = float(np.sum(cr))
    cum_e = float(np.sum(er))
    excess_vs_csi = cum_p - cum_c
    excess_vs_ew = cum_p - cum_e

    # Sharpe
    excess_d = pr - cr
    sharpe = float(np.mean(excess_d) / (np.std(excess_d) + 1e-8) * np.sqrt(252))

    # Max drawdown
    cum_series = np.cumsum(pr)
    peak = np.maximum.accumulate(cum_series)
    max_dd = float(np.max(peak - cum_series))

    # Win rate vs csi
    daily_win = float(np.mean(pr > cr))

    return {
        "N_hold": N_hold, "sell_k": sell_k,
        "cum_return": round(cum_p, 4), "csi300_cum": round(cum_c, 4),
        "ew_cum": round(cum_e, 4),
        "excess_csi": round(excess_vs_csi, 4), "excess_ew": round(excess_vs_ew, 4),
        "sharpe": round(sharpe, 4), "max_drawdown": round(max_dd, 4),
        "daily_win": round(daily_win, 4), "trade_days_pct": round(trades / max(valid_days, 1), 4),
        "avg_pos": round(np.mean(pos_history), 4) if pos_history else 1.0,
        "days": valid_days,
    }

# --- Run sweep ---
print(f"\n{'='*75}")
print(f" Strategy Sweep (with Strategy B stop-loss)")
print(f"{'='*75}")

results = []
for N in N_HOLDS:
    for K in SELL_KS:
        if K >= N:
            continue
        r = backtest(N, K)
        if r:
            results.append(r)
            print(f"  N={N:>2} K={K:>2} | cum={r['cum_return']:+6.2f}% "
                  f"ex_csi={r['excess_csi']:+6.2f}% ex_ew={r['excess_ew']:+6.2f}% "
                  f"sharpe={r['sharpe']:.3f} dd={r['max_drawdown']:.1f}% "
                  f"win={r['daily_win']:.1%}")

df_res = pd.DataFrame(results).sort_values("excess_ew", ascending=False)
df_res.to_csv(OUT_PATH, index=False)

print(f"\n{'='*75}")
best = df_res.iloc[0]
print(f" BEST vs EW:  N={best['N_hold']}, sell_k={best['sell_k']} -> ex_ew={best['excess_ew']:+6.3f}%")
best_csi = df_res.sort_values("excess_csi", ascending=False).iloc[0]
print(f" BEST vs CSI: N={best_csi['N_hold']}, sell_k={best_csi['sell_k']} -> ex_csi={best_csi['excess_csi']:+6.3f}%")

print(f"\n Top 5 by excess vs equal-weight:")
print(f"{'N':>4} {'K':>4} {'CumRet':>8} {'ExcessEW':>9} {'ExcessCSI':>9} {'Sharpe':>7} {'MaxDD':>7} {'WinCSI':>7}")
print("-" * 70)
for _, r in df_res.head(5).iterrows():
    print(f"{r['N_hold']:>4.0f} {r['sell_k']:>4.0f} {r['cum_return']:>+7.2f}% "
          f"{r['excess_ew']:>+8.2f}% {r['excess_csi']:>+8.2f}% "
          f"{r['sharpe']:>7.3f} {r['max_drawdown']:>6.1f}% "
          f"{r['daily_win']:>6.1%}")

print(f"\n[sweep] Done -> {OUT_PATH}")
