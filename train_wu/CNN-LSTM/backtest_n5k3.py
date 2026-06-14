"""
CNN-LSTM 模型 N=5, K=3 回测（统一评测框架）

基于 maxmin_cnn_lstm_L20.pt 模型，加载测试数据生成预测，
然后运行 N=5, K=3 旋转策略回测，采用净值法交易成本。

用法:
  python backtest_n5k3.py
"""
import sys, json
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from stock_predictor.config import RESULT_DIR
from stock_predictor.data import load_scale_frame, build_splits
from stock_predictor.trainer import load_checkpoint, predict_arrays

# ========== 策略参数 ==========
N_FULL = 5
N_RISK = 4
K = 3
CSI5D_THRESH = -0.01
SELL_COST = 0.00076
BUY_COST = 0.00026
TEST_START = 20260105
TEST_END = 20260529


def compute_csi5d_map():
    """读取 CSI300 数据，预计算每个交易日的 5 日累积收益。"""
    csi_path = PROJECT_ROOT.parent.parent / "train_li" / "data" / "market" / "000300.SH.csv"
    if not csi_path.exists():
        print("[warn] CSI300 数据不存在，将不使用风控策略")
        return {}

    csi = pd.read_csv(csi_path, dtype={"trade_date": str})
    csi = csi.sort_values("trade_date")
    csi["pct_chg"] = pd.to_numeric(csi["pct_chg"], errors="coerce") / 100
    csi5d = {}
    for i in range(len(csi)):
        if i < 4:
            csi5d[csi.iloc[i]["trade_date"]] = 0.0
        else:
            csi5d[csi.iloc[i]["trade_date"]] = csi.iloc[i - 4: i + 1]["pct_chg"].sum()
    return csi5d


def run_nk_backtest(signals, csi5d_map):
    """N=5, K=3 旋转回测，净值法交易成本。

    沿用 strategy.py 的现金+持仓回测逻辑，但加入净值法交易成本。
    """
    signals = signals.copy()
    # 过滤无效值
    signals = signals[np.isfinite(signals["real_ret"])].copy()
    signals = signals.sort_values(
        ["signal_date", "pred_ret"], ascending=[True, False]
    ).reset_index(drop=True)

    # 按信号日分组
    date_groups = signals.groupby("signal_date")
    dates = sorted(date_groups.groups.keys())

    # 只保留测试期内的日期
    dates = [d for d in dates if TEST_START <= d <= TEST_END]
    print(f"[backtest] 测试期: {dates[0]} ~ {dates[-1]}, 共 {len(dates)} 个交易日")

    # holdings = set of ts_code（等权，不追踪实际股数）
    holdings = set()
    daily_records = []
    total_trade_cost = 0.0

    # 预先构建设置 real_ret 索引用于查上一日的收益率
    # real_ret_by_date[signal_date][ts_code] = real_ret
    real_ret_map = {}
    for d in dates:
        g = date_groups.get_group(d)
        real_ret_map[d] = g.set_index("ts_code")["real_ret"].to_dict()

    for i, signal_date in enumerate(dates):
        g = date_groups.get_group(signal_date)
        g = g.sort_values("pred_ret", ascending=False).reset_index(drop=True)
        all_stocks = g["ts_code"].tolist()
        stock_set = set(all_stocks)

        # 风控判断
        csi5d = csi5d_map.get(str(signal_date), 0.0)
        risk_off = csi5d < CSI5D_THRESH
        target_n = N_RISK if risk_off else N_FULL
        desired = set(all_stocks[:target_n])

        # 计算当日收益：等权持仓收益（使用前一日信号日的 real_ret）
        port_ret = 0.0
        if holdings:
            prev_date = dates[i - 1] if i > 0 else None
            if prev_date and prev_date in real_ret_map:
                prev_rets = real_ret_map[prev_date]
                held = [c for c in holdings if c in prev_rets]
                if held:
                    port_ret = float(np.mean([prev_rets[c] for c in held]))

        # 换手决策
        if i == 0:
            # 首日：直接持有所选股票
            holdings = desired.copy()
            n_sell, n_buy = 0, len(holdings)
        else:
            # 已有持仓：决定卖哪些
            held_ranked = sorted(
                [(c, g.loc[g["ts_code"] == c, "pred_ret"].values[0] if (g["ts_code"] == c).any() else -999)
                 for c in holdings],
                key=lambda x: x[1]
            )
            n_held = len(holdings)
            n_to_sell = max(0, n_held - target_n)  # 风控降仓
            n_to_sell = max(n_to_sell, min(K, n_held))  # 至少轮换
            n_sell = min(n_to_sell, n_held)
            sell_set = set(c for c, _ in held_ranked[:n_sell])
            holdings -= sell_set

            # 买入
            candidates = [c for c in all_stocks if c not in holdings]
            n_buy = target_n - len(holdings)
            if n_buy > 0:
                holdings.update(candidates[:n_buy])

        # 交易成本
        cost_sell = n_sell * SELL_COST / max(target_n, 1)
        cost_buy = n_buy * BUY_COST / max(target_n, 1)
        trade_cost = cost_sell + cost_buy
        total_trade_cost += trade_cost

        daily_records.append({
            "signal_date": int(signal_date),
            "csi5d": round(csi5d * 100, 2),
            "risk_off": risk_off,
            "n_hold": len(holdings),
            "port_ret": round(port_ret * 100, 4),
            "trade_cost": round(trade_cost * 100, 4),
        })

        if (i + 1) % 30 == 0 or i == 0 or i == len(dates) - 1:
            flag = " [RISK OFF]" if risk_off else ""
            print(f"  D+{i+1:2d} {signal_date} | hold={len(holdings):1d} "
                  f"ret={port_ret*100:+.2f}% cost={trade_cost*100:.3f}%{flag}")

    result = pd.DataFrame(daily_records)
    return result, total_trade_cost


def compute_summary(result, total_cost):
    rets = result["port_ret"].values / 100
    cum = float(np.cumprod(1 + rets)[-1] - 1) if len(rets) > 0 else 0.0
    n_days = len(rets)
    annual_ret = (1 + cum) ** (252 / max(n_days, 1)) - 1
    annual_vol = float(np.std(rets) * np.sqrt(252))
    sharpe = (float(np.mean(rets)) * 252) / annual_vol if annual_vol > 0 else 0.0

    peak = np.maximum.accumulate(np.cumprod(1 + rets))
    dd = np.cumprod(1 + rets) / peak - 1
    max_dd = float(dd.min())
    win_rate = float(np.mean(rets > 0))

    print(f"\n{'='*50}")
    print(f"  CNN-LSTM N={N_FULL}, K={K} 回测结果（等权法）")
    print(f"{'='*50}")
    print(f"  测试周期:    {result['signal_date'].iloc[0]} ~ {result['signal_date'].iloc[-1]}")
    print(f"  交易日数:    {n_days}")
    print(f"  累积收益:    {cum*100:+.2f}%")
    print(f"  年化收益:    {annual_ret*100:+.2f}%")
    print(f"  年化波动:    {annual_vol*100:.2f}%")
    print(f"  Sharpe:      {sharpe:.2f}")
    print(f"  最大回撤:    {max_dd*100:.2f}%")
    print(f"  日胜率:      {win_rate*100:.1f}%")
    print(f"  总交易成本:  {total_cost*100:.2f}%")
    print(f"  风控触发率:  {result['risk_off'].mean()*100:.1f}%")
    print(f"  日均持仓:    {result['n_hold'].mean():.1f} 只")

    return {
        "test_period": f"{result['signal_date'].iloc[0]} ~ {result['signal_date'].iloc[-1]}",
        "trading_days": n_days,
        "cumulative_return": round(cum * 100, 2),
        "annual_return": round(annual_ret * 100, 2),
        "annual_vol": round(annual_vol * 100, 2),
        "sharpe": round(sharpe, 2),
        "max_drawdown": round(max_dd * 100, 2),
        "win_rate": round(win_rate * 100, 1),
        "total_trade_cost": round(total_cost * 100, 2),
        "risk_off_rate": round(result["risk_off"].mean() * 100, 1),
        "avg_hold": round(result["n_hold"].mean(), 1),
    }


def main():
    model_path = PROJECT_ROOT / "result" / "maxmin_cnn_lstm_L20.pt"
    print(f"[load] 加载模型: {model_path}")
    model, cfg, device = load_checkpoint(model_path, device_name="cpu")
    model = model.to(device)
    model.eval()
    print(f"  模型: {cfg.model_name}, seq_len={cfg.seq_len}, scale={cfg.scale_name}")

    print("[load] 加载数据（2024年起）...")
    df = load_scale_frame(cfg.scale_name, year=2024)
    print(f"  数据: {len(df)} 行, {df['ts_code'].nunique()} 只股票")

    print("[data] 构建滑动窗口 ...")
    train_arr, val_arr, test_arr = build_splits(df, scale_name=cfg.scale_name, seq_len=cfg.seq_len)
    print(f"  训练: {len(train_arr.x)}, 验证: {len(val_arr.x)}, 测试: {len(test_arr.x)}")

    print("[infer] 模型预测 ...")
    signals = predict_arrays(model, test_arr, batch_size=4096, device=device, scale_name=cfg.scale_name)
    print(f"  完成: {len(signals)} 条, 日期: {signals['signal_date'].min()} ~ {signals['signal_date'].max()}")

    print("\n[backtest] 加载 CSI300 风控数据 ...")
    csi5d = compute_csi5d_map()

    print("[backtest] 运行 N=5, K=3 回测 ...")
    result_df, total_cost = run_nk_backtest(signals, csi5d)

    print("\n[summary] 汇总 ...")
    summary = compute_summary(result_df, total_cost)

    # 保存
    out_dir = RESULT_DIR / "backtest"
    out_dir.mkdir(parents=True, exist_ok=True)
    result_df.to_csv(out_dir / "backtest_n5k3_daily.csv", index=False)
    with open(out_dir / "backtest_n5k3_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f"\n  明细: {out_dir / 'backtest_n5k3_daily.csv'}")
    print(f"  汇总: {out_dir / 'backtest_n5k3_summary.json'}")


if __name__ == "__main__":
    main()
