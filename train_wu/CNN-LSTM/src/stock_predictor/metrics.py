from __future__ import annotations

import numpy as np
import pandas as pd


PERIOD_DAYS = 12


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    err = y_pred - y_true
    mse = float(np.mean(err**2))
    mae = float(np.mean(np.abs(err)))
    rmse = float(np.sqrt(mse))
    ss_res = float(np.sum(err**2))
    ss_tot = float(np.sum((y_true - float(np.mean(y_true))) ** 2))
    r2 = float("nan") if ss_tot == 0 else float(1.0 - ss_res / ss_tot)
    return {"mse": mse, "mae": mae, "rmse": rmse, "r2": r2}


def direction_win_rate(pred_ret: np.ndarray, real_ret: np.ndarray) -> float:
    mask = np.isfinite(pred_ret) & np.isfinite(real_ret) & (real_ret != 0)
    if not np.any(mask):
        return float("nan")
    return float(np.mean(np.sign(pred_ret[mask]) == np.sign(real_ret[mask])))


def fill_missing_with_mean(df: pd.DataFrame, numeric_cols: list[str]) -> pd.DataFrame:
    """检查缺失值，并用列均值填充。"""

    out = df.copy()
    for col in numeric_cols:
        if col not in out.columns:
            continue
        ser = out[col]
        if not ser.isna().any():
            continue
        mean_val = float(ser.mean(skipna=True))
        if not np.isfinite(mean_val):
            mean_val = 0.0
        out[col] = ser.fillna(mean_val)
    return out


def add_period_id(df: pd.DataFrame, period_days: int = PERIOD_DAYS) -> pd.DataFrame:
    """把连续交易日按 period_days 分段。"""

    out = df.copy()
    uniq_dates = np.sort(out["signal_date"].drop_duplicates().to_numpy(dtype=np.int32, copy=False))
    date_to_period = {int(d): int(i // period_days) for i, d in enumerate(uniq_dates.tolist())}
    out["period_id"] = out["signal_date"].map(date_to_period).astype(np.int32)
    return out


def regression_metrics_by_period(signals: pd.DataFrame, true_col: str, pred_col: str, period_days: int = PERIOD_DAYS) -> tuple[pd.DataFrame, dict[str, float]]:
    filled = fill_missing_with_mean(signals, [true_col, pred_col])
    filled = add_period_id(filled, period_days=period_days)
    rows: list[dict[str, float]] = []
    for pid, g in filled.groupby("period_id", sort=True):
        metrics = regression_metrics(
            g[true_col].to_numpy(dtype=np.float64, copy=False),
            g[pred_col].to_numpy(dtype=np.float64, copy=False),
        )
        metrics["period_id"] = int(pid)
        metrics["start_date"] = int(g["signal_date"].min())
        metrics["end_date"] = int(g["signal_date"].max())
        rows.append(metrics)

    period_df = pd.DataFrame(rows)
    if period_df.empty:
        return period_df, {"mse": float("nan"), "mae": float("nan"), "rmse": float("nan"), "r2": float("nan")}
    summary = {
        "mse": float(period_df["mse"].mean()),
        "mae": float(period_df["mae"].mean()),
        "rmse": float(period_df["rmse"].mean()),
        "r2": float(period_df["r2"].mean()),
    }
    return period_df, summary


def direction_win_rate_by_period(signals: pd.DataFrame, period_days: int = PERIOD_DAYS) -> tuple[pd.DataFrame, dict[str, float]]:
    filled = fill_missing_with_mean(signals, ["pred_ret", "real_ret"])
    filled = add_period_id(filled, period_days=period_days)
    rows: list[dict[str, float]] = []
    for pid, g in filled.groupby("period_id", sort=True):
        win_rate = direction_win_rate(
            g["pred_ret"].to_numpy(dtype=np.float64, copy=False),
            g["real_ret"].to_numpy(dtype=np.float64, copy=False),
        )
        rows.append(
            {
                "period_id": int(pid),
                "start_date": int(g["signal_date"].min()),
                "end_date": int(g["signal_date"].max()),
                "direction_win_rate": float(win_rate),
            }
        )
    period_df = pd.DataFrame(rows)
    summary = {"direction_win_rate": float(period_df["direction_win_rate"].mean())} if not period_df.empty else {"direction_win_rate": float("nan")}
    return period_df, summary


def daily_ic_summary(signals: pd.DataFrame, period_days: int = PERIOD_DAYS) -> tuple[pd.DataFrame, dict[str, float]]:
    filled = fill_missing_with_mean(signals, ["pred_ret", "real_ret"])
    filled = add_period_id(filled, period_days=period_days)
    rows: list[dict[str, float]] = []
    for pid, g in filled.groupby("period_id", sort=True):
        if g.shape[0] < 3:
            continue
        x = g["pred_ret"].to_numpy(dtype=np.float64, copy=False)
        y = g["real_ret"].to_numpy(dtype=np.float64, copy=False)
        if np.std(x) == 0 or np.std(y) == 0:
            continue
        ic = float(np.corrcoef(x, y)[0, 1])
        rank_ic = float(pd.Series(x).corr(pd.Series(y), method="spearman"))
        rows.append(
            {
                "period_id": int(pid),
                "start_date": int(g["signal_date"].min()),
                "end_date": int(g["signal_date"].max()),
                "ic": ic,
                "rank_ic": rank_ic,
            }
        )

    daily = pd.DataFrame(rows)
    if daily.empty:
        return daily, {"ic": float("nan"), "ir": float("nan"), "rank_ic": float("nan"), "rank_ir": float("nan")}

    ic_mean = float(daily["ic"].mean())
    ic_std = float(daily["ic"].std(ddof=0))
    rank_mean = float(daily["rank_ic"].mean())
    rank_std = float(daily["rank_ic"].std(ddof=0))
    summary = {
        "ic": ic_mean,
        "ir": float("nan") if ic_std == 0 else float(ic_mean / ic_std),
        "rank_ic": rank_mean,
        "rank_ir": float("nan") if rank_std == 0 else float(rank_mean / rank_std),
    }
    return daily, summary
