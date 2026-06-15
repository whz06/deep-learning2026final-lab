#!/usr/bin/env python3
"""
统一评测回测 — CNN-LSTM 模型
================================
严格对标 plot_report.py 中的 run_nk_backtest 逻辑。

数据流:
  1. 加载原始 maxmin 缩放 CSV（与训练时同源）
  2. build_splits 按每只股票 80%/20% 切分（与训练时同源）
  3. predict_arrays 输出 pred_ret 和 real_ret（均为小数，如 0.166951 = +16.70%）
  4. 转换格式后喂入回测函数

两个回测函数:
  - run_nk_backtest: 无成本、无风控（纯看模型预测能力）
  - bt_cost: 含交易成本 + Strategy B 风控（统一评测标准）

收益对齐:
  T+1 执行，用 prev_dr 保存前一日的 real_ret。
  第 i 日的收益 = 第 i-1 日选中持仓在第 i-1 日的 real_ret 均值
  = 从第 i-1 日收盘到第 i 日收盘的实际收益
"""
import os, sys, numpy as np, pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
WU_ROOT = ROOT
sys.path.insert(0, str(os.path.join(WU_ROOT, "src")))
from stock_predictor.data import load_scale_frame, build_splits
from stock_predictor.trainer import load_checkpoint, predict_arrays

# ====== 1. 加载模型 ======
model_path = os.path.join(WU_ROOT, "result", "maxmin_cnn_lstm_L20.pt")
print(f"[load] model: {model_path}")
model, cfg, device = load_checkpoint(model_path, device_name="cpu")
model = model.to(device).eval()

# ====== 2. 加载原始 CSV ======
# 不传 year，用全部数据构建窗口（与训练时同源）
df = load_scale_frame(cfg.scale_name)
print(f"  {len(df)} rows, {df['ts_code'].nunique()} stocks")

# ====== 3. 构建滑动窗口 ======
_, _, test_arr = build_splits(df, scale_name=cfg.scale_name, seq_len=cfg.seq_len)
print(f"  test: {len(test_arr.x)} windows, {len(set(test_arr.ts_code))} stocks")
print(f"  dates: {test_arr.signal_date.min()} ~ {test_arr.signal_date.max()}")

# ====== 4. 模型推理 ======
# predict_arrays 输出:
#   - pred_ret: 预测收益(小数), 如 0.166951 = +16.6951%
#   - real_ret: 实际收益(小数), 如 0.200032 = +20.0032%
signals = predict_arrays(model, test_arr, batch_size=4096, device=device, scale_name=cfg.scale_name)
print(f"  {len(signals)} rows, {signals['ts_code'].nunique()} stocks, "
      f"dates {signals['signal_date'].min()} ~ {signals['signal_date'].max()}")

# ====== 5. 回测函数 A: 无成本版 ======
def run_nk_backtest(scores_df, returns_df, n_hold=5, k_rotate=3):
    """
    无成本、无风控的 N-K 旋转回测。

    参数:
      scores_df:  columns=[trade_date, ts_code, score]
                 score = pred_ret（小数）
      returns_df: columns=[trade_date, ts_code, pct_chg]
                 pct_chg = real_ret（小数）

    逻辑:
      - D 日按 pred_ret 取 top N 作为目标持仓
      - 已有持仓中 pred_ret 最低的 K 只卖出
      - 非持仓中 pred_ret 最高的 K 只买入
      - 每日收益 = 持仓股 real_ret 的等权均值
      - 首日无先前持仓，收益记为 0
    """
    dates = sorted(scores_df["trade_date"].unique())
    if len(dates) < 2:
        return pd.DataFrame()
    # 构建 pred_ret 和 real_ret 的日期索引字典
    sl, rl = {}, {}
    for d in dates:
        sl[d] = scores_df[scores_df["trade_date"]==d].set_index("ts_code")["score"]
        rl[d] = returns_df[returns_df["trade_date"]==d].set_index("ts_code")["pct_chg"]
    held, rows = set(), []
    for i, sd in enumerate(dates[:-1]):
        ds = sl.get(sd, pd.Series(dtype=float))
        dr = rl.get(sd, pd.Series(dtype=float))
        if len(ds) < n_hold: continue
        if i == 0:
            # 首日：直接买 top N
            tn = set(ds.nlargest(n_hold).index) & set(dr.index)
            pr = dr[list(tn)].mean() if tn else 0.0
            held = tn
        else:
            # 持仓股收益 = 上一日选中股在当前日的 real_ret
            hv = held & set(dr.index)
            pr = dr[list(hv)].mean() if hv else 0.0
            # 换手
            tn = set(ds.nlargest(n_hold).index) & set(dr.index)
            hr = sorted(hv, key=lambda x: ds.get(x,-1e9))
            ts = set(hr[:k_rotate]) if len(hr)>=k_rotate else set(hr)
            ts &= hv
            cb = (tn-hv)
            tb = set(sorted(cb, key=lambda x: ds.get(x,-1e9), reverse=True)[:k_rotate])
            held = (hv-ts)|tb
        rows.append({"signal_date":sd,"port_ret":pr})
    return pd.DataFrame(rows)

# ====== 6. 转换数据格式 ======
# signals 中的 pred_ret 和 real_ret 已在 decimal 格式
scores_df = signals[["signal_date","ts_code","pred_ret"]].rename(
    columns={"signal_date":"trade_date","pred_ret":"score"})
returns_df = signals[["signal_date","ts_code","real_ret"]].rename(
    columns={"signal_date":"trade_date","real_ret":"pct_chg"})

print(f"\n  scores: {len(scores_df)} rows, {scores_df['trade_date'].nunique()} dates")
print(f"  returns: {len(returns_df)} rows, {returns_df['trade_date'].nunique()} dates")

# ====== 7. 只保留 2026 年测试期（与统一评测对齐）======
scores_df = scores_df[scores_df["trade_date"] >= 20260105]
returns_df = returns_df[returns_df["trade_date"] >= 20260105]

# ====== 8. 无成本回测 ======
print("\n[backtest] N=5 K=3 (no cost, no Strategy B) ...")
eq = run_nk_backtest(scores_df, returns_df, n_hold=5, k_rotate=3)
print(f"  {len(eq)} trading days")
cum = (1+eq["port_ret"]).prod()
ann = cum**(252/len(eq))-1
sp = eq["port_ret"].mean()/(eq["port_ret"].std()+1e-8)*np.sqrt(252)
dd = (1+eq["port_ret"]).cumprod(); mdd=(dd/dd.cummax()-1).min()
print(f"  累积收益:  {(cum-1)*100:+.2f}%")
print(f"  年化收益:  {ann*100:+.2f}%")
print(f"  Sharpe:    {sp:.2f}")
print(f"  最大回撤:  {mdd*100:.2f}%")
print(f"  日胜率:    {(eq['port_ret']>0).mean()*100:.1f}%")

# Rank IC 计算
print("\n[IC] daily RankIC ...")
merged = scores_df.merge(returns_df, on=["trade_date","ts_code"], suffixes=("","_r"))
daily_ic = merged.groupby("trade_date").apply(
    lambda g: g["score"].rank().corr(g["pct_chg"].rank()) if len(g)>=10 else None
).dropna()
print(f"  RankIC: mean={daily_ic.mean():.4f} std={daily_ic.std():.4f} >0:{(daily_ic>0).mean()*100:.0f}%")

# ====== 9. 回测函数 B: 含成本 + Strategy B ======
def bt_cost(scores_df, returns_df, csi5d_map, n_hold=5, k_rotate=3):
    """
    含交易成本和 Strategy B 风控的 N-K 旋转回测。

    与 run_nk_backtest 的区别:
      1. 交易成本: 卖 0.076% + 买 0.026%
      2. Strategy B: CSI5d < -1% 时持仓降至 80%
      3. 收益对齐: 用 prev_dr 保存前一日的 real_ret

    收益对齐详解:
      i=0 (首日):
        - 按当日 pred_ret 选 top N, 买入
        - prev_dr = 当日 real_ret (保存, 下次用)
        - pr = 0 (无先前持仓)

      i=1 (第二日):
        - hv = i=0 选中的股票 ∩ prev_dr(第0日 real_ret) 的股票集合
        - pr = hv 在 prev_dr 中的均值
              = 第0日选中股票在第0日的 real_ret 均值
              = 从第0日收盘到第1日收盘的实际收益  ✓
        - 换手后, prev_dr = 第1日的 real_ret

      i=2 (第三日):
        - hv = 新持仓 ∩ prev_dr(第1日 real_ret)
        - pr = hv 在 prev_dr 中的均值
              = 从第1日收盘到第2日收盘的实际收益  ✓

    Parameters:
      scores_df:  [trade_date, ts_code, score]  score = pred_ret (decimal)
      returns_df: [trade_date, ts_code, pct_chg] pct_chg = real_ret (decimal)
      csi5d_map:  {trade_date: csi5d_value} CSI300 5日累积收益
    """
    dates = sorted(scores_df["trade_date"].unique())
    if len(dates)<2: return pd.DataFrame()

    # 构建日期索引的查询字典
    # sl[D] = {ts_code: pred_ret} 全部股票在 D 日的预测收益
    # rl[D] = {ts_code: real_ret} 全部股票在 D 日的实际收益
    # 注意: pred_ret 和 real_ret 都是 decimal（非百分比）
    sl, rl = {}, {}
    for d in dates:
        sl[d] = scores_df[scores_df["trade_date"]==d].set_index("ts_code")["score"]
        rl[d] = returns_df[returns_df["trade_date"]==d].set_index("ts_code")["pct_chg"]

    held, rows, prev_dr = set(), [], None  # prev_dr 暂存前一日 real_ret
    for i, sd in enumerate(dates[:-1]):
        ds = sl.get(sd, pd.Series(dtype=float))  # sd 日的 pred_ret
        dr = rl.get(sd, pd.Series(dtype=float))   # sd 日的 real_ret
        if len(ds)<n_hold: continue

        # Strategy B: CSI5d < -1% 触发风控
        c5 = csi5d_map.get(str(sd), 0)
        ro = c5<-1.0
        nt = max(1,int(n_hold*0.8)) if ro else n_hold  # 风控时 = 4, 正常 = 5

        if i==0:
            # 首日：无先前持仓
            tn = set(ds.nlargest(nt).index)&set(dr.index)  # top nt by pred_ret
            prev_dr = dr           # 保存当日 real_ret 给下次用
            held = tn              # 当日选中持仓
            cost = (0.00076+0.00026)*nt/n_hold  # 首日全仓买入成本
            pr = 0.0               # 无先前持仓，收益为 0
        else:
            # 持仓股收益 = prev_dr 中这些股票的 real_ret
            # prev_dr 保存的是上一日的 real_ret
            hv = held&set(prev_dr.index)
            pr = prev_dr[list(hv)].mean() if hv else 0.0

            # 换手: 卖 K 最差，买 K 最好
            tn = set(ds.nlargest(nt).index)&set(dr.index)  # 当日 top nt
            hr = sorted(hv, key=lambda x: ds.get(x,-1e9))  # 持仓按当日 pred_ret 排序
            ts = set(hr[:k_rotate]) if len(hr)>=k_rotate else set(hr)
            ts &= hv
            cb = (tn-hv)
            tb = set(sorted(cb, key=lambda x: ds.get(x,-1e9), reverse=True)[:k_rotate])
            nt2 = max(len(ts),len(tb))
            cost = (0.00076+0.00026)*nt2/n_hold  # 换手成本
            held = (hv-ts)|tb                     # 新持仓
            prev_dr = dr                          # 保存当日 real_ret 给下次用

        rows.append({"signal_date":sd,"port_ret":pr-cost,"n_hold":nt,"risk_off":ro})
    return pd.DataFrame(rows)


# ====== CSI300 风控数据 ======
csi_path = os.path.join(ROOT, "..", "..", "train_li", "data", "market", "000300.SH.csv")
csi = pd.read_csv(csi_path, dtype={"trade_date":str}).sort_values("trade_date")
csi["pct_chg"] = pd.to_numeric(csi["pct_chg"], errors="coerce")
csi5d = {}
for i in range(4,len(csi)):
    csi5d[csi.iloc[i]["trade_date"]] = csi.iloc[i-4:i+1]["pct_chg"].sum()

# ====== 10. 含成本回测 ======
print("\n[backtest] N=5 K=3 (with cost sell=0.076% buy=0.026%) ...")
eq2 = bt_cost(scores_df, returns_df, csi5d, n_hold=5, k_rotate=3)
cum2 = (1+eq2["port_ret"]).prod()
sp2 = eq2["port_ret"].mean()/(eq2["port_ret"].std()+1e-8)*np.sqrt(252)
dd2 = (1+eq2["port_ret"]).cumprod(); mdd2=(dd2/dd2.cummax()-1).min()
print(f"  累积收益:  {(cum2-1)*100:+.2f}%")
print(f"  Sharpe:    {sp2:.2f}")
print(f"  最大回撤:  {mdd2*100:.2f}%")
print(f"  日胜率:    {(eq2['port_ret']>0).mean()*100:.1f}%")

# ====== 11. 保存结果 ======
# port_ret 目前是 decimal，plot_report.py 期望百分比值
eq2_out = eq2.copy()
eq2_out["port_ret"] = eq2_out["port_ret"] * 100.0  # 小数→百分比
# 保存到 train_wu 路径（plot_report.py 的 WU_DAILY 路径）
wu_csv = os.path.join(WU_ROOT, "result", "backtest", "backtest_n5k3_daily.csv")
os.makedirs(os.path.dirname(wu_csv), exist_ok=True)
eq2_out.to_csv(wu_csv, index=False)
print(f"\n[save] Wu backtest saved to {wu_csv}")

# 同时也保存到 report 的 results 目录
rpt_csv = os.path.join(ROOT, "..", "..", "train_li", "report", "results", "backtest_n5k3_daily.csv")
os.makedirs(os.path.dirname(rpt_csv), exist_ok=True)
eq2.to_csv(rpt_csv, index=False)
print(f"[save] also saved to {rpt_csv}")
