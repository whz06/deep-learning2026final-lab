"""v8/validate_rotation.py — 行业轮动择时回测

用 v7 Spatial 现有 daily_scores，对比：
  A: Strategy B  (csi5d < -1% → 80% 仓位)
  B: Strategy B+ (行业轮动信号决定仓位)

行业轮动信号：防御型行业平均涨跌 - 周期型行业平均涨跌
"""
import os, sys, glob, numpy as np, pandas as pd

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(SCRIPT_DIR)
SCORES_PATH = os.path.join(ROOT, "v7", "results", "daily_scores_spatial_t1.parquet")
BASIC_PATH = os.path.join(ROOT, "data", "basic.csv")
INDEX_PATH = os.path.join(ROOT, "data", "market", "000300.SH.csv")
PARQUET_PATH = os.path.join(ROOT, "processed", "all_data.parquet")

N_HOLD, K_SELL = 5, 3

# ===== Industry groups =====
DEFENSIVE = [
    "银行", "全国地产", "区域地产", "食品", "医疗保健",
    "化学制药", "生物制药", "中成药", "家用电器",
    "文教休闲", "商业百货", "酒店餐饮",
]

CYCLICAL = [
    "半导体", "电气设备", "证券", "小金属", "汽车配件",
    "化工原料", "专用机械", "元器件", "通信设备",
    "软件服务", "互联网", "航空", "铜", "铝", "钢加工",
    "染料涂料", "塑料", "机械基件",
]

# Rotation thresholds for position sizing
# rotation > 0 → defense stronger (risk-off)
ROT_WEAK, ROT_STRONG = 0.5, 2.0   # % points
POS_WEAK, POS_STRONG, POS_NORMAL = 0.60, 0.40, 1.0

# Strategy B original
CSI5D_THRESH, RISK_OFF = -1.0, 0.80

# ===== Load data =====
print("[rot] Loading v7 Spatial scores ...")
scores = pd.read_parquet(SCORES_PATH)
print(f"[rot] Scores: {len(scores)} records")

print("[rot] Loading industry & CSI300 ...")
basic = pd.read_csv(BASIC_PATH, dtype={"ts_code": str})
basic["industry"] = basic["industry"].fillna("Other")
ts2ind = dict(zip(basic["ts_code"], basic["industry"]))
default_ind = "Other"

# Classify each industry
all_ind = set(basic["industry"].unique())
def_set = set(DEFENSIVE); cyc_set = set(CYCLICAL)
ind_class = {}
for ind in all_ind:
    if ind in def_set: ind_class[ind] = "defensive"
    elif ind in cyc_set: ind_class[ind] = "cyclical"
    else: ind_class[ind] = "neutral"

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
print("[rot] Loading returns ...")
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
print(f"[rot] Test: {test_dates[0]} ~ {test_dates[-1]}, {len(test_dates)} days")

# ===== Compute daily rotation signal =====
print("[rot] Computing rotation signal ...")
rotation_signal = {}
for d in test_dates:
    if d not in ret_d: continue
    rr = ret_d[d]
    def_ret, cyc_ret = [], []
    for ts, ret_val in rr.items():
        ind = ts2ind.get(ts, default_ind)
        if ind in def_set: def_ret.append(ret_val)
        elif ind in cyc_set: cyc_ret.append(ret_val)
    if def_ret and cyc_ret:
        rotation_signal[d] = np.mean(def_ret) - np.mean(cyc_ret)
    else:
        rotation_signal[d] = 0.0

# 5-day EMA of rotation signal
ema_rotation = {}
vals = [rotation_signal.get(d, 0.0) for d in test_dates]
alpha = 2.0 / 6.0  # span=5
ema = vals[0]
for i, d in enumerate(test_dates):
    if i > 0:
        ema = alpha * rotation_signal.get(d, 0.0) + (1 - alpha) * ema
    ema_rotation[d] = ema

print(f"  Rotation signal range: {min(ema_rotation.values()):+.3f} ~ {max(ema_rotation.values()):+.3f}")

# ===== Backtest engine =====
def backtest(strategy_label, use_rotation):
    holdings = None
    daily_p, daily_c = [], []
    position_log = []  # (date, position, trigger_reason)
    false_signals = 0
    true_signals = 0

    for si in range(len(test_dates) - 1):
        fd = test_dates[si]; rd = test_dates[si + 1]
        if fd not in sd or rd not in ret_d: continue
        ss = sd[fd]; rr = ret_d[rd]

        sorted_s = sorted(ss.items(), key=lambda x: x[1], reverse=True)
        top = [c for c, _ in sorted_s if c in rr]
        if len(top) < N_HOLD: continue

        # Position sizing
        if use_rotation:
            rot = ema_rotation.get(fd, 0.0)
            if rot >= ROT_STRONG:
                pos, trigger = POS_STRONG, f"rot_strong({rot:+.2f}%)"
            elif rot >= ROT_WEAK:
                pos, trigger = POS_WEAK, f"rot_weak({rot:+.2f}%)"
            else:
                pos, trigger = POS_NORMAL, "normal"
        else:
            csi5d = get_csi5d(fd)
            if csi5d < CSI5D_THRESH:
                pos, trigger = RISK_OFF, f"csi5d({csi5d:+.2f}%)"
            else:
                pos, trigger = 1.0, "normal"

        target = max(1, int(round(N_HOLD * pos)))
        position_log.append((fd, pos, trigger))

        if holdings is None:
            holdings = top[:target]
        else:
            held_s = [(c, ss.get(c, -1e6)) for c in holdings if c in ss]
            held_s.sort(key=lambda x: x[1], reverse=True)
            to_sell = {c for c, _ in held_s[-K_SELL:]} if len(held_s) > K_SELL else set()
            extra = len(holdings) - target
            if extra > 0:
                cand = [c for c, _ in held_s if c not in to_sell]
                for c in cand[-extra:]: to_sell.add(c)
            holdings = [c for c in holdings if c not in to_sell]
            held_set = set(holdings)
            for c in top:
                if len(holdings) >= target: break
                if c not in held_set: holdings.append(c)

        pr = np.mean([rr.get(c, 0.0) for c in holdings]) if holdings else 0.0
        cr = csi_map.get(rd, 0.0)
        daily_p.append(pr); daily_c.append(cr)

    if not daily_p: return None

    # Analyze signal quality
    for j in range(len(position_log)):
        pdate, pos, trigger = position_log[j]
        if pos < 1.0:  # risk-off signal
            # Check next 5-day CSI300 return
            idx_in_td = test_dates.index(pdate)
            next_5 = [test_dates[idx_in_td + k] for k in range(1, 6)
                       if idx_in_td + k < len(test_dates)]
            fut_ret = sum(csi_map.get(d, 0.0) for d in next_5)
            if fut_ret < 0:
                true_signals += 1
            else:
                false_signals += 1

    cum = np.sum(daily_p); cum_c = np.sum(daily_c)
    daily_arr = np.array(daily_p)
    sharpe = np.mean(daily_arr) / (np.std(daily_arr, ddof=1) + 1e-12) * np.sqrt(252)

    # Risk-off days
    risk_off_days = sum(1 for _, p, _ in position_log if p < 1.0)
    total_days = len(position_log)

    result = {
        "strategy": strategy_label,
        "cum": round(cum, 4), "excess": round(cum - cum_c, 4),
        "sharpe": round(sharpe, 3), "days": len(daily_p),
        "risk_off_days": risk_off_days,
        "risk_off_pct": round(risk_off_days / max(total_days, 1) * 100, 1),
        "true_signals": true_signals, "false_signals": false_signals,
        "signal_accuracy": f"{true_signals}/{true_signals + false_signals}"
            if (true_signals + false_signals) > 0 else "N/A",
    }
    return result, daily_p, daily_c, position_log


# ===== Run =====
print(f"\n{'='*70}")
print(f" 行业轮动择时 — N={N_HOLD} K={K_SELL}")
print(f" 防御行业: {len(def_set)} 个 | 周期行业: {len(cyc_set)} 个")
print(f" 阈值: weak={ROT_WEAK}% → {POS_WEAK:.0%}, strong={ROT_STRONG}% → {POS_STRONG:.0%}")
print(f"{'='*70}")

r_csi, dp_csi, dc_csi, pl_csi = backtest("Strategy B (csi5d)", False)
r_rot, dp_rot, dc_rot, pl_rot = backtest("Strategy B+ (rotation)", True)

if r_csi and r_rot:
    print(f"\n{'Strategy':<25} {'Cum':>8} {'Excess':>8} {'Sharpe':>7} {'RiskOff':>8} {'Off%':>6} {'True/False'}")
    print("-" * 80)
    for r in [r_csi, r_rot]:
        print(f"{r['strategy']:<25} {r['cum']:>+7.2f}% {r['excess']:>+7.2f}% {r['sharpe']:>7.3f} "
              f"{r['risk_off_days']:>8} {r['risk_off_pct']:>5.1f}% {r['signal_accuracy']:>10}")

    delta = r_rot["cum"] - r_csi["cum"]
    print(f"\n  轮动择时增量: {delta:+.2f}% cumulative")
    print(f"  信号准确率: csi5d={r_csi['signal_accuracy']}, rotation={r_rot['signal_accuracy']}")

    # Print rotation signal at specific dates
    print(f"\n  信号示例（前15天）:")
    for d, pos, trigger in pl_rot[:15]:
        rot = ema_rotation.get(d, 0.0)
        csi5 = get_csi5d(d)
        print(f"    {d}  rotation={rot:+.2f}%  csi5d={csi5:+.2f}%  pos={pos:.0%}  [{trigger}]")

print(f"\n[rot] Done.")
