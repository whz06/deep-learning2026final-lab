from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from .config import HOLD_N, INITIAL_CASH, LOT_SIZE, RESULT_DIR, ROTATE_K


@dataclass
class Position:
    shares: int
    buy_date: int


def top_picks_report(signals: pd.DataFrame, top_k: int, last_n_days: int, out_path: Path) -> Path:
    """生成最近 N 天的 TopK 建仓建议报告。"""

    out_path.parent.mkdir(parents=True, exist_ok=True)
    dates = np.sort(signals["signal_date"].drop_duplicates().to_numpy())
    selected = dates[-last_n_days:] if dates.size >= last_n_days else dates
    lines = [f"# 最近 {len(selected)} 个交易日的 Top{top_k} 建仓建议", ""]

    for d in selected.tolist():
        g = signals[signals["signal_date"] == int(d)].sort_values("pred_ret", ascending=False, kind="mergesort").head(top_k).copy()
        lines.append(f"## 信号日 {int(d)}")
        lines.append("")
        lines.append("| 排名 | 股票 | 现价 | 预测下一日价格 | 预测收益率(%) |")
        lines.append("|---:|---|---:|---:|---:|")
        for i, (_, row) in enumerate(g.iterrows(), start=1):
            lines.append(
                f"| {i} | {row['ts_code']} | {float(row['cur_close']):.4f} | {float(row['pred_raw']):.4f} | {float(row['pred_ret']) * 100:.3f} |"
            )
        lines.append("")

    out_path.write_text("\n".join(lines), encoding="utf-8")
    return out_path


def run_backtest(
    signals: pd.DataFrame,
    n: int = HOLD_N,
    k: int = ROTATE_K,
    initial_cash: float = INITIAL_CASH,
    lot_size: int = LOT_SIZE,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, float]]:
    """按 N-K、T+1、一手 100 股规则回测。"""

    signals = signals[np.isfinite(signals["real_ret"].to_numpy())].copy()
    signals = signals.sort_values(["signal_date", "pred_ret"], ascending=[True, False], kind="mergesort").reset_index(drop=True)

    cash = float(initial_cash)
    positions: dict[str, Position] = {}
    daily_rows: list[dict[str, float]] = []
    holding_rows: list[dict[str, float]] = []

    for di, (signal_date, g) in enumerate(signals.groupby("signal_date", sort=True)):
        target_date = int(g["target_date"].iloc[0])
        cur_price = g.set_index("ts_code")["cur_close"].astype(np.float64)
        pred_ret = g.set_index("ts_code")["pred_ret"].astype(np.float64)
        real_ret = g.set_index("ts_code")["real_ret"].astype(np.float64)

        desired = g["ts_code"].head(n).astype(str).tolist()
        desired_set = set(desired)
        held_set = set(positions.keys())

        if di == 0:
            sell_list: list[str] = []
            buy_list = desired
        else:
            sell_candidates = [code for code in held_set - desired_set if positions[code].buy_date < int(signal_date)]
            buy_candidates = [code for code in desired_set - held_set]
            sell_candidates.sort(key=lambda code: float(pred_ret.get(code, -1e9)))
            buy_candidates.sort(key=lambda code: float(pred_ret.get(code, -1e9)), reverse=True)
            trade_n = min(k, len(sell_candidates), len(buy_candidates))
            sell_list = sell_candidates[:trade_n]
            buy_list = buy_candidates[:trade_n]

        traded_amount = 0.0
        for code in sell_list:
            if code not in cur_price.index:
                continue
            price = float(cur_price[code])
            shares = int(positions[code].shares)
            cash += shares * price
            traded_amount += shares * price
            del positions[code]

        after_sell_value = cash + sum(int(pos.shares) * float(cur_price.get(code, 0.0)) for code, pos in positions.items())
        target_value = after_sell_value / float(max(1, n))

        for code in buy_list:
            if code not in cur_price.index:
                continue
            price = float(cur_price[code])
            shares = int((min(target_value, cash) // (price * lot_size)) * lot_size)
            if shares <= 0:
                continue
            need_cash = shares * price
            cash -= need_cash
            traded_amount += need_cash
            positions[code] = Position(shares=shares, buy_date=int(signal_date))

        hold_value = sum(int(pos.shares) * float(cur_price.get(code, 0.0)) for code, pos in positions.items())
        total_value = cash + hold_value
        next_value = cash + sum(int(pos.shares) * float(cur_price[code]) * (1.0 + float(real_ret.get(code, 0.0))) for code, pos in positions.items() if code in cur_price.index)
        daily_ret = 0.0 if total_value <= 0 else next_value / total_value - 1.0

        daily_rows.append(
            {
                "signal_date": int(signal_date),
                "target_date": target_date,
                "cash": cash,
                "hold_value": hold_value,
                "total_value": total_value,
                "daily_ret": daily_ret,
                "n_hold": len(positions),
                "sell_n": len(sell_list),
                "buy_n": len(buy_list),
                "traded_amount": traded_amount,
            }
        )

        for code, pos in positions.items():
            holding_rows.append(
                {
                    "signal_date": int(signal_date),
                    "ts_code": code,
                    "shares": int(pos.shares),
                    "buy_date": int(pos.buy_date),
                    "price": float(cur_price.get(code, np.nan)),
                    "value": int(pos.shares) * float(cur_price.get(code, 0.0)),
                    "pred_ret": float(pred_ret.get(code, np.nan)),
                }
            )

    daily = pd.DataFrame(daily_rows)
    holdings = pd.DataFrame(holding_rows)
    daily["equity"] = daily["total_value"] / float(initial_cash)
    drawdown = daily["equity"] / daily["equity"].cummax() - 1.0
    annual_return = float(daily["equity"].iloc[-1] ** (252.0 / max(1, daily.shape[0])) - 1.0)
    sharpe = float(np.mean(daily["daily_ret"]) / np.std(daily["daily_ret"], ddof=0) * np.sqrt(252.0)) if daily["daily_ret"].std(ddof=0) > 0 else float("nan")
    summary = {
        "annual_return": annual_return,
        "max_drawdown": float(drawdown.min()),
        "sharpe": sharpe,
        "final_value": float(daily["total_value"].iloc[-1]),
        "n_days": int(daily.shape[0]),
    }
    return daily, holdings, summary


def save_backtest(prefix: str, daily: pd.DataFrame, holdings: pd.DataFrame, summary: dict[str, float], result_dir: Path | None = None) -> dict[str, Path]:
    result_dir = RESULT_DIR / "backtest" if result_dir is None else result_dir
    result_dir.mkdir(parents=True, exist_ok=True)

    daily_path = result_dir / f"{prefix}_daily.csv"
    holding_path = result_dir / f"{prefix}_holdings.csv.gz"
    summary_path = result_dir / f"{prefix}_summary.json"

    daily.to_csv(daily_path, index=False, encoding="utf-8")
    holdings.to_csv(holding_path, index=False, encoding="utf-8")
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return {"daily": daily_path, "holdings": holding_path, "summary": summary_path}

