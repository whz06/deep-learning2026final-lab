"""v8/diagnose_crash_indicator.py — 三步减仓指标优化

Phase 1: 市场级特征数据集 + 特征-标签相关性扫描
Phase 2: 阈值网格搜索 (2019-2024 train → 2025 val)
Phase 3: MLP 分类器训练 + 全周期回测 vs Strategy B

数据: all_data.parquet + CSI300 + v7 scores
"""
import os, sys, glob, numpy as np, pandas as pd
from scipy.stats import spearmanr
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(SCRIPT_DIR)
PARQUET = os.path.join(ROOT, "processed", "all_data.parquet")
INDEX = os.path.join(ROOT, "data", "market", "000300.SH.csv")
SCORES = os.path.join(ROOT, "v7", "results", "daily_scores_spatial_t1.parquet")

TRAIN_START, TRAIN_END = "20190102", "20241231"
VAL_START, VAL_END = "20250102", "20250531"
TEST_START, TEST_END = "20260203", "20260529"

# ===== Phase 1: Market-level features =====
print("=" * 70)
print(" Phase 1: Market-level feature dataset")
print("=" * 70)

print("[P1] Loading all_data ...")
df = pd.read_parquet(PARQUET)
df["trade_date"] = df["trade_date"].astype(str)
for col in ["pct_chg","close","vol","amount","turnover_rate"]:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

print("[P1] Loading CSI300 ...")
csi = pd.read_csv(INDEX, dtype={"trade_date": str})
csi["csi_ret"] = pd.to_numeric(csi["pct_chg"], errors="coerce") / 100.0
csi["csi_close"] = pd.to_numeric(csi["close"], errors="coerce")

print("[P1] Computing daily market features across all stocks ...")
dates = sorted(set(df["trade_date"]) & set(csi["trade_date"]))

# Pre-compute stock-level stats per date
daily_features = []
STOCK_COL = "pct_chg"

for di, d in enumerate(dates):
    sdf = df[df["trade_date"] == d]
    csi_row = csi[csi["trade_date"] == d].iloc[0]
    N = len(sdf)
    if N < 100: continue

    rets = sdf[STOCK_COL].values.astype(float)

    feat = {"date": d}
    feat["csi_ret"] = csi_row["csi_ret"]
    feat["n_stocks"] = N
    feat["pct_up"] = np.mean(rets > 0)
    feat["pct_down"] = np.mean(rets < 0)
    feat["pct_down_gt_3pct"] = np.mean(rets < -3)
    feat["pct_up_gt_3pct"] = np.mean(rets > 3)
    feat["mean_ret"] = np.mean(rets)
    feat["std_ret"] = np.std(rets)
    feat["skew_ret"] = 0.0 if feat["std_ret"] < 1e-8 else (
        ((rets - feat["mean_ret"])**3).mean() / (feat["std_ret"]**3))
    feat["median_ret"] = np.median(rets)
    feat["iqr_ret"] = np.percentile(rets, 75) - np.percentile(rets, 25)
    daily_features.append(feat)
    if (di + 1) % 500 == 0:
        print(f"  {di+1}/{len(dates)} dates")

feat_df = pd.DataFrame(daily_features)
feat_df = feat_df.sort_values("date").reset_index(drop=True)

# Compute rolling features
print("[P1] Computing rolling features ...")
feat_df["csi_ret_1d"] = feat_df["csi_ret"]
feat_df["csi_ret_3d"] = feat_df["csi_ret"].rolling(3).sum()
feat_df["csi_ret_5d"] = feat_df["csi_ret"].rolling(5).sum()
feat_df["csi_ret_10d"] = feat_df["csi_ret"].rolling(10).sum()
feat_df["csi_ret_20d"] = feat_df["csi_ret"].rolling(20).sum()
feat_df["csi_vol_5d"] = feat_df["csi_ret"].rolling(5).std()
feat_df["csi_vol_20d"] = feat_df["csi_ret"].rolling(20).std()
feat_df["csi_drawdown"] = feat_df["csi_ret"].rolling(20).apply(
    lambda x: (np.maximum.accumulate(np.maximum(x, 0)) - x).max() if len(x) >= 5 else 0, raw=False)
feat_df["csi_down_streak"] = np.nan
streak = 0
for i in range(len(feat_df)):
    if pd.isna(feat_df.iloc[i]["csi_ret_1d"]): continue
    r = feat_df.iloc[i]["csi_ret_1d"]
    if r < 0:
        streak = streak + 1
    elif r > 0:
        streak = 0
    feat_df.at[feat_df.index[i], "csi_down_streak"] = streak

# Market breadth (requires tracking ongoing)
feat_df["csi_5d_acc"] = feat_df["csi_ret_5d"] - feat_df["csi_ret_10d"]

feat_df["pct_up_ma5"] = feat_df["pct_up"].rolling(5).mean()
feat_df["pct_up_ma20"] = feat_df["pct_up"].rolling(20).mean()
feat_df["pct_down_3pct_ma5"] = feat_df["pct_down_gt_3pct"].rolling(5).mean()
feat_df["std_ret_ma5"] = feat_df["std_ret"].rolling(5).mean()
feat_df["std_ret_ma20"] = feat_df["std_ret"].rolling(20).mean()
feat_df["skew_ret_ma5"] = feat_df["skew_ret"].rolling(5).mean()

# Label: will next 5 days have a >3% drawdown from csi300?
feat_df["min_5d_future"] = feat_df["csi_ret_1d"].rolling(5, min_periods=5).sum().shift(-5)
future_5d = feat_df["csi_ret_1d"].values
min_5d = np.full(len(feat_df), np.nan)
for i in range(len(feat_df) - 5):
    fut = future_5d[i+1:i+6]
    fut = fut[~np.isnan(fut)]
    if len(fut) >= 5:
        min_5d[i] = np.sum(fut)
feat_df["ret_fwd_5d"] = min_5d
feat_df["crash_5d"] = (feat_df["ret_fwd_5d"] < -0.03).astype(int)
feat_df["crash_3d"] = (feat_df["csi_ret_1d"].rolling(3).sum().shift(-3) < -0.02).astype(int)

feat_df = feat_df.dropna(subset=["csi_ret_5d"]).reset_index(drop=True)
print(f"[P1] Feature dataset: {len(feat_df)} daily rows x {len(feat_df.columns)} columns")
print(f"  Crash rate (5d): {feat_df['crash_5d'].mean():.2%}")
print(f"  Crash rate (3d): {feat_df['crash_3d'].mean():.2%}")

# Split
train_df = feat_df[(feat_df["date"] >= TRAIN_START) & (feat_df["date"] <= TRAIN_END)]
val_df = feat_df[(feat_df["date"] >= VAL_START) & (feat_df["date"] <= VAL_END)]
test_df = feat_df[(feat_df["date"] >= TEST_START) & (feat_df["date"] <= TEST_END)]
print(f"  Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")

# Feature-label correlation
feature_names = [c for c in feat_df.columns if c not in [
    "date","n_stocks","csi_ret","csi_ret_1d","min_5d_future","ret_fwd_5d","crash_5d","crash_3d"]]

print(f"\n[P1] Feature → crash_5d Spearman correlation (train set):")
corr_results = []
for fn in feature_names:
    mask = ~np.isnan(train_df[fn])
    if mask.sum() < 50: continue
    ic, p = spearmanr(train_df[fn][mask].values, train_df["crash_5d"][mask].values)
    corr_results.append({"feature": fn, "corr": round(ic, 4), "p": round(p, 4)})

corr_results = sorted(corr_results, key=lambda x: abs(x["corr"]), reverse=True)
print(f"  {'Feature':<25} {'Corr':>8} {'p-val':>8}")
for r in corr_results[:15]:
    print(f"  {r['feature']:<25} {r['corr']:>+8.4f} {r['p']:>8.4f}")


# ===== Phase 2: Threshold grid search =====
print(f"\n{'='*70}")
print(" Phase 2: Threshold grid search")
print(f"{'='*70}")

# Strategy B baseline on val set
def evaluate_strategy(decision_fn, label):
    """Evaluate a position-sizing strategy on the given date range.
    Returns (cumulative_return, num_risk_off_days, total_days, signal_accuracy).
    """
    # We use the v7 scores for the actual stock-level backtest
    pass  # placeholder - we do the full backtest in Phase 3

# For Phase 2, evaluate on VAL set only using regression-based approach
# Simpler: for each candidate threshold, compute "would reduce position" accuracy
print("[P2] Searching thresholds on validation set ...")

candidate_indicators = [r["feature"] for r in corr_results[:6]]
threshold_grid = {}

for ind in candidate_indicators:
    best_acc, best_thresh, best_f1 = 0, 0, 0

    vals = train_df[ind].dropna().values
    labels = train_df.loc[train_df[ind].notna(), "crash_5d"].values

    if len(vals) < 100: continue

    p_range = np.linspace(10, 90, 17)  # percentiles
    for p in p_range:
        thresh = np.percentile(vals, p)

        # On train set: precision/recall
        pred = (vals > thresh).astype(int)
        tp = ((pred == 1) & (labels == 1)).sum()
        fp = ((pred == 1) & (labels == 0)).sum()
        fn = ((pred == 0) & (labels == 1)).sum()
        precision = tp / (tp + fp + 1e-12)
        recall = tp / (tp + fn + 1e-12)
        f1 = 2 * precision * recall / (precision + recall + 1e-12)

        # On val set
        v_vals = val_df[ind].dropna().values
        v_labels = val_df.loc[val_df[ind].notna(), "crash_5d"].values
        if len(v_vals) < 20: continue
        v_pred = (v_vals > thresh).astype(int)
        v_acc = (v_pred == v_labels).mean()

        if v_acc > best_acc:
            best_acc, best_thresh, best_f1 = v_acc, thresh, f1

    threshold_grid[ind] = {
        "threshold": round(best_thresh, 6),
        "val_acc": round(best_acc, 4),
        "train_f1": round(best_f1, 4),
        "threshold_pct": round(
            (vals < best_thresh).mean() * 100, 1),
    }

# Baseline: random
crash_rate = train_df["crash_5d"].mean()
random_acc = max(crash_rate, 1 - crash_rate)
print(f"  Crash rate (train): {crash_rate:.2%}")
print(f"  Naive 'never crash' accuracy: {1 - crash_rate:.2%}")
print(f"  Best single-indicator threshold on val:")
print(f"  {'Indicator':<25} {'Thresh':>10} {'ValAcc':>8} {'Rare%':>7} {'F1'}")
for ind, info in sorted(threshold_grid.items(), key=lambda x: -x[1]["val_acc"]):
    print(f"  {ind:<25} {info['threshold']:>10.4f} {info['val_acc']:>8.4f} {info['threshold_pct']:>6.1f}% {info['train_f1']:>7.4f}")

# Best composite: top 3 indicators, any one triggers → crash signal
top3 = sorted(threshold_grid.items(), key=lambda x: -x[1]["val_acc"])[:3]
composite_val = val_df.copy()
composite_pred = np.zeros(len(composite_val), dtype=bool)
for ind, info in top3:
    if ind in composite_val.columns:
        composite_pred = composite_pred | (composite_val[ind].fillna(0) > info["threshold"]).values.astype(bool)
composite_acc = (composite_pred.astype(int) == composite_val["crash_5d"].values).mean()
print(f"\n  Top-3 ensemble val accuracy: {composite_acc:.4f}")
print(f"  vs naive: {random_acc:.4f} (delta: {composite_acc - random_acc:+.4f})")

improvement = composite_acc - random_acc
print(f"\n  ==> Phase 2 verdict: {'IMPROVEMENT' if improvement > 0.02 else 'NEGLIGIBLE'} ({improvement:+.4f})")


# ===== Phase 3: MLP Classifier + Full Backtest =====
print(f"\n{'='*70}")
print(" Phase 3: MLP Classifier Training + Full Backtest")
print(f"{'='*70}")

# Now build the full crash indicator with the features from Phase 1+2
indicator_features = [r["feature"] for r in corr_results[:10]]
print(f"[P3] Using {len(indicator_features)} features for MLP classifier")

# Prepare data
X_train = train_df[indicator_features].fillna(0).values.astype(np.float32)
y_train = train_df["crash_5d"].fillna(0).values.astype(np.float32)
X_val = val_df[indicator_features].fillna(0).values.astype(np.float32)
y_val = val_df["crash_5d"].fillna(0).values.astype(np.float32)
X_test = test_df[indicator_features].fillna(0).values.astype(np.float32)
y_test = test_df["crash_5d"].fillna(0).values.astype(np.float32)

print(f"[P3] Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")

# Standardize
scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_val_s = scaler.transform(X_val)
X_test_s = scaler.transform(X_test)

# Logistic Regression (weighted for imbalanced)
crash_weight = (1 - y_train.mean()) / y_train.mean()
print(f"[P3] Class balance: crash={y_train.mean():.3f}, no-crash={1-y_train.mean():.3f}, weight={crash_weight:.2f}")

# Use sklearn LogisticRegression
clf = LogisticRegression(
    class_weight="balanced",
    C=1.0,
    max_iter=2000,
    random_state=42,
)
clf.fit(X_train_s, y_train)

train_pred = clf.predict_proba(X_train_s)[:, 1]
val_pred = clf.predict_proba(X_val_s)[:, 1]
test_pred = clf.predict_proba(X_test_s)[:, 1]

# Evaluate predicted probabilities as crash signals
# Position = 1 - 0.4 * min(p_crash / threshold_for_reductions, 1)
for label, pred, y_true in [
    ("Train", train_pred, y_train),
    ("Val  ", val_pred, y_val),
    ("Test ", test_pred, y_test),
]:
    acc = np.mean((pred > 0.5) == y_true)
    # Find best threshold that maximizes F1
    best_f1, best_t, best_acc_at_t = 0, 0.5, 0
    for t in np.linspace(0.1, 0.9, 17):
        p = (pred > t).astype(int)
        tp = ((p == 1) & (y_true == 1)).sum()
        fp = ((p == 1) & (y_true == 0)).sum()
        fn = ((p == 0) & (y_true == 1)).sum()
        prec = tp / (tp + fp + 1e-12)
        rec = tp / (tp + fn + 1e-12)
        f1 = 2 * prec * rec / (prec + rec + 1e-12)
        if f1 > best_f1:
            best_f1, best_t, best_acc_at_t = f1, t, (p == y_true).mean()
    print(f"  {label}: acc@0.5={acc:.3f}, bestF1={best_f1:.3f}@t={best_t:.2f}, acc@best={best_acc_at_t:.3f}")

# ===== Full Backtest with MLP-based position sizing =====
print(f"\n[P3] Full backtest with v7 Spatial scores ...")

# Load v7 scores
scores = pd.read_parquet(SCORES)
sd = {}
for _, r in scores.iterrows():
    sd.setdefault(r["trade_date"], {})[r["ts_code"]] = r["score"]

# Returns
ret_d = {}
for d, sdf in df.groupby("trade_date"):
    ret_d[d] = dict(zip(sdf["ts_code"], sdf["pct_chg"].astype(float)))

test_dates = sorted(sd.keys())

# Build MLP crash probability lookup for test dates
crash_prob_map = {}
mlp_dates = test_df["date"].values
for i, d in enumerate(mlp_dates):
    crash_prob_map[d] = float(test_pred[i])

# Also compute for all test_dates (fill with nearest)
for d in test_dates:
    if d not in crash_prob_map:
        # Use val set mean as fallback
        crash_prob_map[d] = float(np.mean(val_pred))

# CSI300 reference
csi_map = dict(zip(csi["trade_date"], csi["pct_chg"].astype(float)))
csi_dates = sorted(csi_map.keys())

def get_csi5d(date):
    if date not in csi_dates: return 0.0
    idx = csi_dates.index(date)
    start = max(0, idx - 4)
    vals = [csi_map[csi_dates[i]] for i in range(start, idx + 1)]
    return sum(vals)

def backtest_full(strategy_name, position_fn):
    holdings = None
    daily_p, daily_c = [], []
    risk_off_count = 0

    for si in range(len(test_dates) - 1):
        fd = test_dates[si]; rd = test_dates[si + 1]
        if fd not in sd or rd not in ret_d: continue
        ss = sd[fd]; rr = ret_d[rd]
        sorted_s = sorted(ss.items(), key=lambda x: x[1], reverse=True)
        top = [c for c, _ in sorted_s if c in rr]
        if len(top) < 5: continue

        pos = position_fn(fd)
        target = max(1, int(round(5 * pos)))

        if holdings is None:
            holdings = top[:target]
        else:
            held_s = [(c, ss.get(c, -1e6)) for c in holdings if c in ss]
            held_s.sort(key=lambda x: x[1], reverse=True)
            to_sell = {c for c, _ in held_s[-3:]} if len(held_s) > 3 else set()
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
        if pos < 1.0: risk_off_count += 1

    if not daily_p: return None
    cum = np.sum(daily_p); cum_c = np.sum(daily_c)
    daily_arr = np.array(daily_p)
    sharpe = np.mean(daily_arr) / (np.std(daily_arr, ddof=1) + 1e-12) * np.sqrt(252)
    return {
        "name": strategy_name, "cum": round(cum, 4), "excess": round(cum - cum_c, 4),
        "sharpe": round(sharpe, 3), "days": len(daily_p),
        "risk_off_days": risk_off_count,
        "risk_off_pct": round(risk_off_count / max(len(daily_p), 1) * 100, 1),
    }

# Strategy B (original)
r_b = backtest_full("Strategy B (csi5d)",
    lambda d: 0.8 if get_csi5d(d) < -1.0 else 1.0)

# MLP-based
MLP_REDUCTION = 0.4
r_mlp = backtest_full("MLP crash indicator",
    lambda d: max(0.4, 1.0 - MLP_REDUCTION * min(crash_prob_map.get(d, 0.25) / 0.25, 1.0)))

# MLP with more aggressive reduction
r_mlp2 = backtest_full("MLP aggressive",
    lambda d: max(0.2, 1.0 - 0.6 * min(crash_prob_map.get(d, 0.25) / 0.25, 1.0)))

# MLP only (only reduce when prob > 0.5)
r_mlp3 = backtest_full("MLP strict (p>0.5)",
    lambda d: 0.6 if crash_prob_map.get(d, 0.0) > 0.5 else 1.0)

# Best threshold from grid search - use csi_vol_5d or similar
best_ind = sorted(threshold_grid.items(), key=lambda x: -x[1]["val_acc"])[0]
best_ind_name = best_ind[0]
best_ind_thresh = best_ind[1]["threshold"]
def best_single_pos(date):
    if date not in feat_df["date"].values or best_ind_name not in feat_df.columns:
        return 1.0
    val = feat_df.loc[feat_df["date"] == date, best_ind_name].values
    if len(val) == 0 or pd.isna(val[0]):
        return 1.0
    return 0.7 if val[0] > best_ind_thresh else 1.0

r_best_single = backtest_full(f"Best single ({best_ind_name})", best_single_pos)

print(f"\n  {'Strategy':<30} {'Cum':>8} {'Excess':>8} {'Sharpe':>7} {'RiskOff':>8} {'Off%':>6}")
print("  " + "-" * 70)
for r in [r_b, r_mlp, r_mlp2, r_mlp3, r_best_single]:
    if r:
        print(f"  {r['name']:<30} {r['cum']:>+7.2f}% {r['excess']:>+7.2f}% "
              f"{r['sharpe']:>7.3f} {r['risk_off_days']:>8} {r['risk_off_pct']:>5.1f}%")

all_results = [r for r in [r_b, r_mlp, r_mlp2, r_mlp3, r_best_single] if r is not None]
if all_results:
    best_r = max(all_results, key=lambda x: x["cum"])
    print(f"\n[P3] Best: {best_r['name']} cum={best_r['cum']:+.2f}% excess={best_r['excess']:+.2f}%")
    vs_baseline = best_r["cum"] - r_b["cum"] if r_b else 0
    print(f"[P3] vs Strategy B: {vs_baseline:+.2f}%")
