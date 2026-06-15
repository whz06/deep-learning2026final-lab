"""
统一评测轻量回测 — N=5, K=3

基于 unified_eval 框架思路：
  - 测试期: 2026-02-03 ~ 2026-05-29 (v7 spatial 模型已有分数的 74 个交易日)
  - 股票池: 每日 scores 与 all_data.parquet 的交集
  - 策略: N=5（满仓），风险-off 时 N=4（80%），每日最多换手 K=3
  - 成本: 净值法 卖 0.076% + 买 0.026%
  - 信号: v7 GRU+SpatialAttention (d=32, K=5) 模型每日 score

用法:
  python report/backtest_n5k3.py
"""
import os, sys
import numpy as np
import pandas as pd
from collections import defaultdict

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

SCORES_PATH = os.path.join(ROOT, "v7", "results", "daily_scores_spatial_t1.parquet")
PARQUET_PATH = os.path.join(ROOT, "processed", "all_data.parquet")
CSI_PATH = os.path.join(ROOT, "data", "market", "000300.SH.csv")

# ========== 策略参数 ==========
N_FULL = 5       # 满仓持仓数
N_RISK = 4       # 风险-off 持仓数 (80%)
K = 3            # 每日最大换手
CSI5D_THRESH = -0.01  # CSI5d < -1% 触发风控
SELL_COST = 0.00076   # 卖出成本 0.076%
BUY_COST  = 0.00026   # 买入成本 0.026%


def load_data():
    """加载分数、收益、指数数据。"""
    scores = pd.read_parquet(SCORES_PATH)
    scores["trade_date"] = scores["trade_date"].astype(str)

    ad = pd.read_parquet(PARQUET_PATH, columns=["ts_code", "trade_date", "pct_chg"])
    ad["trade_date"] = ad["trade_date"].astype(str)

    csi = pd.read_csv(CSI_PATH, dtype={"trade_date": str})
    csi = csi.sort_values("trade_date")
    csi["pct_chg"] = pd.to_numeric(csi["pct_chg"], errors="coerce")

    return scores, ad, csi


def compute_csi5d(csi, date):
    """计算截至 date 的 CSI300 5 日累积收益。"""
    idx = csi[csi["trade_date"] == date].index
    if len(idx) == 0:
        return 0.0
    pos = idx[0]
    if pos < 5:
        return 0.0
    recent = csi.iloc[pos - 4: pos + 1]["pct_chg"]
    return recent.sum() / 100  # 转为小数


def backtest(scores, returns, csi):
    """运行 N=5, K=3 回测，返回每日持仓和收益明细。"""
    dates = sorted(scores["trade_date"].unique())
    print(f"[backtest] 共 {len(dates)} 个交易日, "
          f"{scores['ts_code'].nunique()} 只股票")

    # 收益索引: (date, ts_code) -> pct_chg
    ret_idx = returns.dropna(subset=["pct_chg"])
    ret_idx["pct_chg"] = ret_idx["pct_chg"] / 100  # 转小数
    ret_map = ret_idx.set_index(["trade_date", "ts_code"])["pct_chg"].to_dict()

    # CSI5d 预计算
    csi5d_map = {}
    for d in dates:
        csi5d_map[d] = compute_csi5d(csi, d)

    # 逐日回测
    holdings = set()       # 当前持仓 ts_code
    daily_records = []     # 每日记录
    total_trade_cost = 0.0

    for i, date in enumerate(dates):
        # 1. 风控判断（先算，后面要用）
        csi5d = csi5d_map.get(date, 0.0)
        target_n = N_RISK if csi5d < CSI5D_THRESH else N_FULL
        risk_off = (csi5d < CSI5D_THRESH)

        # 2. 获取当日分数 + 取交集（只保留有收益数据的股票）
        day_scores = scores[scores["trade_date"] == date].copy()
        if day_scores.empty:
            continue

        # 显式取交集：只保留当天有收益数据的股票
        day_scores["has_ret"] = day_scores["ts_code"].apply(
            lambda c: ret_map.get((date, c)) is not None
        )
        day_scores = day_scores[day_scores["has_ret"]].drop(columns=["has_ret"])

        if len(day_scores) < target_n:
            continue

        day_scores = day_scores.sort_values("score", ascending=False)
        all_stocks = day_scores["ts_code"].tolist()

        # 3. 计算当日持仓收益
        port_ret = 0.0
        if holdings and i > 0:
            daily_rets = []
            for code in list(holdings):
                r = ret_map.get((date, code))
                if r is not None and np.isfinite(r):
                    daily_rets.append(r)
            port_ret = np.mean(daily_rets) if daily_rets else 0.0

        # 4. 换手
        n_held = len(holdings)
        if n_held == 0:
            # 首日：直接买 top-N
            to_buy = all_stocks[:target_n]
            trade_cost = sum(BUY_COST for _ in to_buy) / max(target_n, 1)
            holdings = set(to_buy)
            trade_qty = len(to_buy)
            total_trade_cost += trade_cost
        else:
            # 已有持仓：决定卖哪些
            held_scores = [(code, day_scores[day_scores["ts_code"] == code]["score"].values)
                           for code in holdings]
            held_ranked = []
            for code in list(holdings):
                sc = day_scores.loc[day_scores["ts_code"] == code, "score"]
                held_ranked.append((code, sc.iloc[0] if len(sc) > 0 else -999))

            held_ranked.sort(key=lambda x: x[1])  # 最差在前

            # 需要卖出的数量
            n_to_sell = max(0, n_held - target_n)  # 风控降仓
            n_to_sell = max(n_to_sell, min(K, n_held))  # 至少轮换 K 只

            # 选出卖出的股票
            sell_candidates = [c for c, _ in held_ranked[:n_to_sell]]
            # 但遵循 K 上限：最多卖 K 只（除非降仓需要更多）
            n_sell = min(n_to_sell, max(K, n_held - target_n))
            sell_candidates = [c for c, _ in held_ranked[:n_sell]]
            sell_set = set(sell_candidates)

            # 执行卖出
            trade_sell_qty = len(sell_set)
            holdings -= sell_set

            # 买入：从非持仓、分数最高的中选
            candidates = [c for c in all_stocks if c not in holdings]
            n_buy = target_n - len(holdings)

            if n_buy > 0:
                to_buy = candidates[:n_buy]
                holdings.update(to_buy)
                trade_buy_qty = len(to_buy)
            else:
                to_buy = []
                trade_buy_qty = 0

            # 交易成本（净值法）
            cost_sell = trade_sell_qty * SELL_COST / max(target_n, 1)
            cost_buy = trade_buy_qty * BUY_COST / max(target_n, 1)
            trade_cost = cost_sell + cost_buy
            total_trade_cost += trade_cost
            trade_qty = trade_sell_qty + trade_buy_qty

        daily_records.append({
            "date": date,
            "csi5d": csi5d,
            "risk_off": risk_off,
            "n_hold": len(holdings),
            "port_ret": port_ret,
            "trade_qty": trade_qty,
            "trade_cost": trade_cost,
            "csi300_ret": csi5d_map.get(date, 0.0),
            "holdings": ",".join(sorted(holdings)),
        })

        if (i + 1) % 20 == 0 or i == 0 or i == len(dates) - 1:
            flag = " [RISK OFF]" if risk_off else ""
            print(f"  D+{i+1:2d} {date} | hold={len(holdings):1d} "
                  f"ret={port_ret*100:+.2f}% cost={trade_cost*100:.3f}%{flag}")

    result = pd.DataFrame(daily_records)
    return result, total_trade_cost


def compute_summary(result, total_trade_cost):
    """计算回测汇总指标。"""
    rets = result["port_ret"].values
    cum = np.cumprod(1 + rets) - 1
    final_cum = cum[-1] if len(cum) > 0 else 0.0

    n_days = len(rets)
    annual_factor = 252 / max(n_days, 1)
    annual_ret = (1 + final_cum) ** annual_factor - 1
    std = np.std(rets) * np.sqrt(252)
    sharpe = (np.mean(rets) * 252) / (np.std(rets) * np.sqrt(252)) if np.std(rets) > 0 else 0.0

    # 最大回撤
    peak = np.maximum.accumulate(np.cumprod(1 + rets))
    dd = np.cumprod(1 + rets) / peak - 1
    max_dd = dd.min()

    # 胜率
    win_rate = np.mean(rets > 0)

    print(f"\n{'='*50}")
    print(f"  统一评测回测结果 — N={N_FULL}, K={K}")
    print(f"{'='*50}")
    print(f"  测试周期:    {result['date'].iloc[0]} ~ {result['date'].iloc[-1]}")
    print(f"  交易日数:    {n_days}")
    print(f"  累积收益:    {final_cum*100:+.2f}%")
    print(f"  年化收益:    {annual_ret*100:+.2f}%")
    print(f"  年化波动:    {std*100:.2f}%")
    print(f"  Sharpe:      {sharpe:.2f}")
    print(f"  最大回撤:    {max_dd*100:.2f}%")
    print(f"  日胜率:      {win_rate*100:.1f}%")
    print(f"  总交易成本:  {total_trade_cost*100:.2f}%")
    print(f"  风控触发率:  {result['risk_off'].mean()*100:.1f}%")
    print(f"  日均持仓:    {result['n_hold'].mean():.1f} 只")

    return {
        "cumulative_return": round(final_cum * 100, 2),
        "annual_return": round(annual_ret * 100, 2),
        "annual_vol": round(std * 100, 2),
        "sharpe": round(sharpe, 2),
        "max_drawdown": round(max_dd * 100, 2),
        "win_rate": round(win_rate * 100, 1),
        "total_trade_cost": round(total_trade_cost * 100, 2),
        "risk_off_rate": round(result["risk_off"].mean() * 100, 1),
        "avg_hold": round(result["n_hold"].mean(), 1),
    }


def main():
    os.makedirs(os.path.join(ROOT, "report", "results"), exist_ok=True)

    print("[load] 加载数据 ...")
    scores, returns, csi = load_data()
    print(f"  scores: {len(scores)} 行, {scores['trade_date'].nunique()} 天")
    print(f"  returns: {len(returns)} 行")
    print(f"  csi300: {len(csi)} 行")

    print("\n[backtest] 运行回测 ...")
    result_df, total_cost = backtest(scores, returns, csi)

    print("\n[summary] 汇总结果 ...")
    summary = compute_summary(result_df, total_cost)

    # 保存
    out_csv = os.path.join(ROOT, "report", "results", "backtest_n5k3_daily.csv")
    out_json = os.path.join(ROOT, "report", "results", "backtest_n5k3_summary.json")
    result_df.to_csv(out_csv, index=False)
    import json
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f"\n  明细已保存: {out_csv}")
    print(f"  汇总已保存: {out_json}")


if __name__ == "__main__":
    main()
