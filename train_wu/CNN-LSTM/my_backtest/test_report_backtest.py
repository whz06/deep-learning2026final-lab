#!/usr/bin/env python3
"""从 all_data.parquet 全量重做 maxmin 缩放 → CNN-LSTM 推理 → 回测"""
import os, sys
import numpy as np
import pandas as pd
from collections import defaultdict

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
WU_ROOT = os.path.join(ROOT, "..", "train_wu", "CNN-LSTM")
sys.path.insert(0, str(os.path.join(WU_ROOT, "src")))
from stock_predictor.models import build_model
from stock_predictor.trainer import load_checkpoint


device = "cpu"

# ── 1. 加载模型 ──
model_path = os.path.join(WU_ROOT, "result", "maxmin_cnn_lstm_L20.pt")
print(f"[load] model: {model_path}", flush=True)
model, cfg, _ = load_checkpoint(model_path, device_name="cpu")
model = model.to(device).eval()

# ── 2. 加载全量行情数据 ──
parquet_path = os.path.join(ROOT, "processed", "all_data.parquet")
print(f"[load] all_data.parquet ...")
all_df = pd.read_parquet(parquet_path, columns=["ts_code", "trade_date", "close", "pct_chg"])
all_df["trade_date"] = all_df["trade_date"].astype(str)
print(f"  {len(all_df)} rows, {all_df['ts_code'].nunique()} stocks, "
      f"{all_df['trade_date'].min()} ~ {all_df['trade_date'].max()}")

# ── 3. 逐股票做渐进式 maxmin 缩放 + 构建窗口 ──
print(f"[data] building windows (seq_len={cfg.seq_len}) ...")
test_start = "20260105"
test_end = "20260529"

fields = ["x", "y_scaled", "y_raw", "cur_close", "next_close",
          "min_ref", "max_ref", "signal_date", "target_date", "ts_code"]
buf = {f: [] for f in fields}

stocks = sorted(all_df["ts_code"].unique())
n_total = len(stocks)

# 只保留 2026 测试期有足够数据的股票
print(f"  筛选有 2026 测试窗口的股票 ...", flush=True)
dates_2026 = sorted(all_df[all_df["trade_date"].between(test_start, test_end)]["trade_date"].unique())
stocks_with_2026 = all_df[all_df["trade_date"].between(test_start, test_end)]["ts_code"].unique()
stocks = sorted(set(stocks) & set(stocks_with_2026))
print(f"  2026 年有数据的股票: {len(stocks)} / {n_total}", flush=True)

SL = cfg.seq_len
progress_step = max(1, len(stocks) // 20)

for si, ts_code in enumerate(stocks):
    sdf = all_df[all_df["ts_code"] == ts_code].sort_values("trade_date").reset_index(drop=True)
    close = sdf["close"].values.astype(np.float64)
    dates = sdf["trade_date"].values.astype(str)
    n = len(close)

    if n < SL + 1:
        continue

    # O(n) 渐进式 maxmin：用累积最大/最小
    cur_max = np.maximum.accumulate(close)
    cur_min = np.minimum.accumulate(close)
    span = cur_max - cur_min
    span[span < 1e-12] = 1.0  # 避免除零
    scaled = (close - cur_min) / span

    # 筛选 2026 测试期的信号日
    sig_dates = dates[SL-1:-1]
    mask = (sig_dates >= test_start) & (sig_dates <= test_end)
    valid_idx = np.where(mask)[0] + (SL-1)  # 映射回 close 的索引

    n_valid = len(valid_idx)
    if n_valid == 0:
        continue

    # 向量化提取 + 追加到 buffer
    for idx in valid_idx:
        i = idx - (SL - 1)  # window start
        buf["x"].append(scaled[i:i+SL].astype(np.float32))
        buf["y_scaled"].append(np.float32(scaled[idx+1]))
        buf["y_raw"].append(np.float32(close[idx+1]))
        buf["cur_close"].append(np.float32(close[idx]))
        buf["next_close"].append(np.float32(close[idx+1]))
        buf["min_ref"].append(np.float32(cur_min[idx]))
        buf["max_ref"].append(np.float32(cur_max[idx]))
        buf["signal_date"].append(dates[idx])
        buf["target_date"].append(dates[idx+1])
        buf["ts_code"].append(ts_code)

    if (si + 1) % progress_step == 0:
        print(f"  {si+1}/{len(stocks)} stocks, {len(buf['x'])} windows", flush=True)

print(f"  total: {len(buf['x'])} windows, {len(set(buf['ts_code']))} stocks, "
      f"{len(set(buf['signal_date']))} dates")

# ── 4. 推理（直接 numpy → torch，不用 WindowArrays）──
print(f"[infer] predicting {len(buf['x'])} samples ...", flush=True)
import torch
from torch.utils.data import DataLoader, TensorDataset

x_arr = np.array(buf["x"], dtype=np.float32)[:, :, np.newaxis]  # [N, 20] → [N, 20, 1]
cur_close_arr = np.array(buf["cur_close"], dtype=np.float32)
min_ref_arr = np.array(buf["min_ref"], dtype=np.float32)
max_ref_arr = np.array(buf["max_ref"], dtype=np.float32)
print(f"  input shape: {x_arr.shape}", flush=True)

dataset = TensorDataset(torch.from_numpy(x_arr).float())
loader = DataLoader(dataset, batch_size=4096, shuffle=False)

preds = []
with torch.no_grad():
    for bx, in loader:
        preds.append(model(bx.to(device)).cpu().numpy())
pred_scaled = np.concatenate(preds)

# 反归一化 + 计算 pred_ret
pred_raw = min_ref_arr + pred_scaled * (max_ref_arr - min_ref_arr)
pred_ret = pred_raw / cur_close_arr - 1.0
next_close_arr = np.array(buf["next_close"], dtype=np.float32)
real_ret = next_close_arr / cur_close_arr - 1.0

signals = pd.DataFrame({
    "signal_date": np.array(buf["signal_date"]).astype(str),
    "ts_code": buf["ts_code"],
    "pred_ret": pred_ret,
    "real_ret": real_ret,
})
valid = np.isfinite(pred_ret) & np.isfinite(real_ret)
signals = signals[valid].reset_index(drop=True)
print(f"  {len(signals)} valid signals", flush=True)

# ── 6. 转成回测函数格式 ──
signals = signals[signals["signal_date"].between(test_start, test_end)]
scores_df = signals[["signal_date", "ts_code", "pred_ret"]].rename(
    columns={"signal_date": "trade_date", "pred_ret": "score"})
returns_df = signals[["signal_date", "ts_code", "real_ret"]].rename(
    columns={"signal_date": "trade_date", "real_ret": "pct_chg"})

print(f"\n  scores_df:  {len(scores_df)} rows, {scores_df['trade_date'].nunique()} dates, "
      f"{scores_df['ts_code'].nunique()} stocks")
print(f"  returns_df: {len(returns_df)} rows, {returns_df['trade_date'].nunique()} dates, "
      f"{returns_df['ts_code'].nunique()} stocks")

# ── 7. CSI5d ──
csi_path = os.path.join(ROOT, "data", "market", "000300.SH.csv")
csi = pd.read_csv(csi_path, dtype={"trade_date": str}).sort_values("trade_date")
csi["pct_chg"] = pd.to_numeric(csi["pct_chg"], errors="coerce")
csi5d = {}
for i in range(4, len(csi)):
    csi5d[csi.iloc[i]["trade_date"]] = csi.iloc[i-4:i+1]["pct_chg"].sum()

# ── 8. 回测 ──
def run_nk_backtest(scores_df, returns_df, csi5d_map, n_hold=5, k_rotate=3):
    dates = sorted(scores_df["trade_date"].unique())
    if len(dates) < 2:
        return pd.DataFrame()
    sl, rl = {}, {}
    for d in dates:
        sl[d] = scores_df[scores_df["trade_date"] == d].set_index("ts_code")["score"]
        rl[d] = returns_df[returns_df["trade_date"] == d].set_index("ts_code")["pct_chg"] / 100.0
    held, rows = set(), []
    for i, sd in enumerate(dates[:-1]):
        nd = dates[i+1]
        ds = sl.get(sd, pd.Series(dtype=float))
        dr = rl.get(sd, pd.Series(dtype=float))
        if len(ds) < n_hold: continue
        c5d = csi5d_map.get(sd, 0)
        ro = c5d < -1.0
        nt = max(1, int(n_hold*0.8)) if ro else n_hold
        if i == 0:
            tn = set(ds.nlargest(nt).index) & set(dr.index)
            pr = dr[list(tn)].mean() if tn else 0.0
            held = tn
            cost = (0.00076+0.00026)*nt/n_hold
        else:
            hv = held & set(dr.index)
            pr = dr[list(hv)].mean() if hv else 0.0
            tn = set(ds.nlargest(nt).index) & set(dr.index)
            hr = sorted(hv, key=lambda x: ds.get(x, -1e9))
            ts = set(hr[:k_rotate]) if len(hr)>=k_rotate else set(hr)
            ts &= hv
            cb = (tn-hv)
            tb = set(sorted(cb, key=lambda x: ds.get(x,-1e9), reverse=True)[:k_rotate])
            nt_ = max(len(ts), len(tb))
            cost = (0.00076+0.00026)*nt_/n_hold
            held = (hv-ts)|tb
        rows.append({"signal_date":sd,"port_ret":pr-cost,"n_hold":nt,"risk_off":ro})
    return pd.DataFrame(rows)

print("\n[backtest] running ...")
eq = run_nk_backtest(scores_df, returns_df, csi5d, n_hold=5, k_rotate=3)
print(f"  {len(eq)} trading days")

cum = (1+eq["port_ret"]).prod()
ann = cum**(252/len(eq))-1
sp = eq["port_ret"].mean()/(eq["port_ret"].std()+1e-8)*np.sqrt(252)
dd = (1+eq["port_ret"]).cumprod(); mdd = (dd/dd.cummax()-1).min()

print(f"\n{'='*50}")
print(f"  CNN-LSTM 全量数据回测 N=5 K=3")
print(f"{'='*50}")
print(f"  每日股票数:  {scores_df.groupby('trade_date').size().describe()}")
print(f"  累积收益:    {(cum-1)*100:+.2f}%")
print(f"  年化收益:    {ann*100:+.2f}%")
print(f"  Sharpe:      {sp:.2f}")
print(f"  最大回撤:    {mdd*100:.2f}%")
print(f"  日胜率:      {(eq['port_ret']>0).mean()*100:.1f}%")
print(f"  风控触发:    {eq['risk_off'].mean()*100:.1f}%")

# 保存
out_dir = os.path.join(WU_ROOT, "result", "backtest")
os.makedirs(out_dir, exist_ok=True)
eq.to_csv(os.path.join(out_dir, "backtest_n5k3_full_daily.csv"), index=False)
print(f"\n  已保存: {out_dir}/backtest_n5k3_full_daily.csv")
