from __future__ import annotations

"""
训练前的数据读取与整理工具。

设计原则：
- 直接接收日频样本表（all_daily_OHLCV_with_indicators.csv）
- 按 trade_date 做截面标准化
- 构建 T 日滑动窗口序列供 LSTM 使用
- 不负责任何旧版月频聚合逻辑
"""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch

RESERVED_COLUMNS: set[str] = {
    "ts_code", "trade_date", "open", "high", "low", "close", "vol", "y",
}

MAX_SEQUENCE_GAP_DAYS: int = 14

PREBUILT_N_FACTORS: int = 26       # v7 pipeline 输出 26 维特征
PREBUILT_TIMESTEP: int = 60        # v7 pipeline T=60


@dataclass(frozen=True)
class PreparedTabularData:
    """训练前整理好的日频表格数据。"""

    frame: pd.DataFrame
    factor_columns: list[str]


@dataclass(frozen=True)
class FactorGanTensors:
    """
    Factor-GAN 需要的张量结果。

    形状说明：
    - z: (N, T, F)，N 条序列，每条 T 日因子窗口
    - y: (N,)，每条序列对应的下一日收益标签
    """

    z: torch.Tensor
    y: torch.Tensor
    meta: pd.DataFrame


def parse_factor_columns_arg(raw: str | None) -> list[str] | None:
    """解析命令行传入的因子列字符串。"""
    if raw is None:
        return None
    items = [token.strip() for token in str(raw).split(",")]
    items = [token for token in items if token]
    return items or None


def load_processed_frame(path: str | Path) -> pd.DataFrame:
    """读取日频样本表。"""
    file_path = Path(path)
    if not file_path.exists():
        raise FileNotFoundError(f"未找到数据文件：{file_path}")

    suffix = file_path.suffix.lower()
    if suffix == ".csv":
        df = pd.read_csv(file_path)
    elif suffix in {".parquet", ".pq"}:
        df = pd.read_parquet(file_path)
    else:
        raise ValueError(f"暂不支持的文件格式：{suffix}。请使用 CSV 或 Parquet。")

    if "ts_code" in df.columns:
        df["ts_code"] = df["ts_code"].astype(str).str.strip().str.upper()
    if "trade_date" in df.columns:
        df["trade_date"] = pd.to_datetime(df["trade_date"])
    return df


def infer_factor_columns(df: pd.DataFrame, factor_columns: list[str] | None = None) -> list[str]:
    """确定本次训练使用哪些因子列。"""
    if factor_columns is not None:
        missing = [col for col in factor_columns if col not in df.columns]
        if missing:
            raise ValueError(f"指定的因子列不存在：{missing}")
        return list(factor_columns)

    inferred: list[str] = []
    for col in df.columns:
        if col in RESERVED_COLUMNS:
            continue
        if pd.api.types.is_numeric_dtype(df[col]):
            inferred.append(col)

    if not inferred:
        raise ValueError("自动推断失败：除保留列外没有发现可作为因子的数值列。")
    return inferred


def add_next_day_return(df: pd.DataFrame) -> pd.DataFrame:
    """
    构造监督学习标签 `y`：下一日收益率。
    y = close.pct_change().shift(-1)，按 ts_code 分组计算。
    """
    out = df.copy()
    out = out.sort_values(["ts_code", "trade_date"])
    out["y"] = out.groupby("ts_code", sort=False)["close"].transform(
        lambda s: s.pct_change().shift(-1)
    )
    return out


def rank_to_unit_interval(
    df: pd.DataFrame,
    factor_columns: list[str],
    group_col: str = "trade_date",
) -> pd.DataFrame:
    """对每个截面的因子做排序标准化，映射到 [-1, 1]。"""

    out = df.copy()

    def _rank_one(series: pd.Series) -> pd.Series:
        pct = series.rank(method="average", pct=True)
        return pct * 2.0 - 1.0

    for col in factor_columns:
        out[col] = out.groupby(group_col, sort=False)[col].transform(_rank_one)
    return out


def prepare_training_frame(
    data_path: str | Path,
    factor_columns: list[str] | None = None,
    group_col: str = "trade_date",
) -> PreparedTabularData:
    """
    读取并整理训练所需的日频样本表。

    步骤：
    1. 读取样本表
    2. 推断因子列
    3. 构造下一日收益标签 y
    4. 逐日截面排序标准化
    5. 丢弃 y 或因子为 NaN 的行
    """
    df = load_processed_frame(data_path)
    columns = infer_factor_columns(df, factor_columns=factor_columns)

    if not columns:
        raise ValueError("因子列不能为空。")

    frame = add_next_day_return(df)
    frame = rank_to_unit_interval(frame, columns, group_col=group_col)
    frame = frame.dropna(subset=["y"] + columns).copy()
    frame = frame.sort_values(["trade_date", "ts_code"]).reset_index(drop=True)
    return PreparedTabularData(frame=frame, factor_columns=columns)


def build_sequences(
    frame: pd.DataFrame,
    factor_columns: list[str],
    timestep: int = 5,
    max_gap_days: int = MAX_SEQUENCE_GAP_DAYS,
) -> FactorGanTensors:
    """
    将日频面板数据转换为滑动窗口序列张量。

    对每只股票，在连续的交易日区间内构建长度为 T 的滑动窗口，
    标签为该窗口最后一日对应的下一日收益 y。

    如果相邻交易日间隔超过 max_gap_days，则断开序列，
    避免跨停牌区间的无效序列。
    """
    if "y" not in frame.columns:
        raise ValueError("输入数据缺少标签列 `y`。")
    missing = [col for col in factor_columns if col not in frame.columns]
    if missing:
        raise ValueError(f"输入数据缺少因子列：{missing}")

    sequences: list[np.ndarray] = []
    labels: list[float] = []
    metas: list[dict] = []

    for _ts_code, group in frame.groupby("ts_code", sort=False):
        group = group.sort_values("trade_date")
        dates = group["trade_date"]
        x = group[factor_columns].to_numpy(dtype=np.float32)
        y_arr = group["y"].to_numpy(dtype=np.float32)

        if len(group) <= timestep:
            continue

        date_diffs = dates.diff().dt.days.fillna(0).values

        for i in range(len(group) - timestep):
            if (date_diffs[i + 1 : i + timestep] > max_gap_days).any():
                continue

            seq = x[i : i + timestep]
            label = float(y_arr[i + timestep - 1])

            if np.isnan(seq).any() or np.isnan(label):
                continue

            sequences.append(seq)
            labels.append(label)
            metas.append({
                "ts_code": group.iloc[i + timestep - 1]["ts_code"],
                "trade_date": dates.iloc[i + timestep - 1],
                "y": label,
            })

    if not sequences:
        raise RuntimeError("未能构建任何有效序列，请检查数据完整性。")

    z = torch.from_numpy(np.stack(sequences))
    y = torch.from_numpy(np.array(labels, dtype=np.float32))
    meta = pd.DataFrame(metas)
    return FactorGanTensors(z=z, y=y, meta=meta)


def load_prebuilt_windows(
    windows_dir: str | Path,
    date_range: list[pd.Timestamp],
) -> FactorGanTensors:
    """
    从 v7 pipeline 输出的预构建 .pt 文件中加载指定日期范围的序列。

    每个 .pt 文件包含:
      - features: (N, T, F) 张量 (F=26, T=60)
      - labels: (N,) 张量
      - ts_codes: list[str]

    参数:
        windows_dir: processed/windows_v7 目录
        date_range: 需要加载的交易日列表

    返回:
        FactorGanTensors(z, y, meta)
    """
    windows_dir = Path(windows_dir)
    all_features: list[torch.Tensor] = []
    all_labels: list[torch.Tensor] = []
    all_metas: list[dict] = []
    loaded_count = 0

    for trade_date in date_range:
        # pd.Timestamp(20190102) → "2019-01-02" → 转为 "20190102" 匹配文件名
        raw = str(trade_date.date()) if hasattr(trade_date, "date") else str(trade_date)
        date_str = raw.replace("-", "")
        year = date_str[:4]
        fname = f"{date_str}.pt"

        # 在 train/val 两个子目录中查找
        pt_path = None
        for split in ("train", "val"):
            candidate = windows_dir / split / year / fname
            if candidate.is_file():
                pt_path = candidate
                break

        if pt_path is None:
            continue

        data = torch.load(pt_path, map_location="cpu", weights_only=True)
        features = data["features"]  # (N, T, F)
        labels = data["labels"]       # (N,)
        ts_codes = data.get("ts_codes", [""] * len(labels))

        all_features.append(features)
        all_labels.append(labels)
        for i, code in enumerate(ts_codes):
            all_metas.append({
                "ts_code": code,
                "trade_date": date_str,
                "y": labels[i].item(),
            })
        loaded_count += 1

    if not all_features:
        raise RuntimeError(
            f"未能从 {windows_dir} 加载任何 .pt 文件 (date_range 中 {len(date_range)} 天, "
            f"范围 {date_range[0]} ~ {date_range[-1]})。请先运行 v7 build_windows 生成数据。"
        )

    z = torch.cat(all_features, dim=0)
    y = torch.cat(all_labels, dim=0)
    meta = pd.DataFrame(all_metas)
    print(f"  [prebuilt] 加载了 {loaded_count} 个日频文件, {len(y)} 条序列, "
          f"特征维度 {z.shape[-1]}, 时间步 {z.shape[1]}")
    return FactorGanTensors(z=z, y=y, meta=meta)
