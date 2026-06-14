from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from numpy.lib.stride_tricks import sliding_window_view

from .config import MAXMIN_CSV, SIGMOID_CSV, RESULT_DIR, TRAIN_RATIO, VAL_RATIO_IN_TRAIN


SIGMOID_START = 0.0
SIGMOID_END = 150.0
SIGMOID_N = abs(SIGMOID_START - SIGMOID_END)
SIGMOID_A = float(np.log(40_000.0))
SIGMOID_B = float(np.log(5e-3))


@dataclass(frozen=True)
class WindowArrays:
    scale_name: str
    x: np.ndarray
    y_scaled: np.ndarray
    y_raw: np.ndarray
    cur_close: np.ndarray
    next_close: np.ndarray
    min_ref: np.ndarray
    max_ref: np.ndarray
    signal_date: np.ndarray
    target_date: np.ndarray
    ts_code: np.ndarray


class PriceWindowDataset(Dataset):
    """把窗口数组包装成 PyTorch 数据集。"""

    def __init__(self, arrays: WindowArrays) -> None:
        self.x = torch.from_numpy(arrays.x)
        self.y_scaled = torch.from_numpy(arrays.y_scaled)
        self.y_raw = torch.from_numpy(arrays.y_raw)
        self.cur_close = torch.from_numpy(arrays.cur_close)
        self.next_close = torch.from_numpy(arrays.next_close)
        self.min_ref = torch.from_numpy(arrays.min_ref)
        self.max_ref = torch.from_numpy(arrays.max_ref)

    def __len__(self) -> int:
        return int(self.x.shape[0])

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, ...]:
        return (
            self.x[idx],
            self.y_scaled[idx],
            self.y_raw[idx],
            self.cur_close[idx],
            self.next_close[idx],
            self.min_ref[idx],
            self.max_ref[idx],
        )


def load_scale_frame(scale_name: str, year: int | None = None) -> pd.DataFrame:
    """按缩放方案读取单独的数据表，可选按年份过滤。"""

    if scale_name == "maxmin":
        df = pd.read_csv(
            MAXMIN_CSV,
            usecols=["ts_code", "trade_date", "close_raw", "min_ref", "max_ref", "close_scaled", "train"],
            dtype={
                "ts_code": str,
                "trade_date": str,
                "close_raw": np.float32,
                "min_ref": np.float32,
                "max_ref": np.float32,
                "close_scaled": np.float32,
                "train": np.int8,
            },
        )
        df = df.rename(columns={"close_scaled": "scaled_value", "train": "is_train"})
        df["ts_code"] = df["ts_code"].astype(str).str.upper()
        df["trade_date"] = df["trade_date"].astype(str).str.replace("-", "", regex=False).astype(np.int32)
        df = df.sort_values(["ts_code", "trade_date"], kind="mergesort").reset_index(drop=True)

    elif scale_name == "sigmoid":
        df = pd.read_csv(
            SIGMOID_CSV,
            usecols=["ts_code", "trade_date", "close", "scaled_close"],
            dtype={"ts_code": str, "trade_date": str, "close": np.float32, "scaled_close": np.float32},
        )
        df = df.rename(columns={"close": "close_raw", "scaled_close": "scaled_value"})
        df["ts_code"] = df["ts_code"].astype(str).str.upper()
        df["trade_date"] = df["trade_date"].astype(str).str.replace("-", "", regex=False).astype(np.int32)
        df = df.sort_values(["ts_code", "trade_date"], kind="mergesort").reset_index(drop=True)
        df["min_ref"] = np.nan
        df["max_ref"] = np.nan
    else:
        raise ValueError(f"未知缩放类型: {scale_name}")

    if year is not None:
        df = df[df["trade_date"] // 10000 >= year].copy()

    df["is_train"] = 0
    for _, g in df.groupby("ts_code", sort=False):
        train_end = int(g.shape[0] * TRAIN_RATIO)
        df.loc[g.index[:train_end], "is_train"] = 1
    return df


def inverse_maxmin(pred_scaled: np.ndarray, min_ref: np.ndarray, max_ref: np.ndarray) -> np.ndarray:
    return pred_scaled * (max_ref - min_ref) + min_ref


def inverse_sigmoid_formula(pred_scaled: np.ndarray) -> np.ndarray:
    """按给定 sigmoid 公式做反变换。

    原始缩放定义中的 score 在 (0, 2) 内，文件里保存的是 score-1，
    因此这里先把 scaled 值平移回 score，再解析求逆得到原始 close。
    """

    score = pred_scaled.astype(np.float64, copy=False) + 1.0
    score = np.clip(score, 1e-6, 2.0 - 1e-6)
    log_term = np.log(2.0 / score - 1.0)
    x = SIGMOID_START + SIGMOID_N - (SIGMOID_N / SIGMOID_A) * (log_term - SIGMOID_B)
    return x.astype(np.float32, copy=False)


def build_splits(
    df: pd.DataFrame,
    scale_name: str,
    seq_len: int,
    train_ratio: float = TRAIN_RATIO,
    val_ratio_in_train: float = VAL_RATIO_IN_TRAIN,
) -> tuple[WindowArrays, WindowArrays, WindowArrays]:
    """构造 train/val/test 三个窗口集合（向量化版本）。"""

    if seq_len <= 0:
        raise ValueError("seq_len 必须大于 0")
    if not (0.0 < train_ratio < 1.0):
        raise ValueError("train_ratio 必须在 (0,1) 内")
    if not (0.0 <= val_ratio_in_train < 1.0):
        raise ValueError("val_ratio_in_train 必须在 [0,1) 内")

    # 每种分片的收集器：list of numpy arrays per stock
    field_names = ["x", "y_scaled", "y_raw", "cur_close", "next_close",
                   "min_ref", "max_ref", "signal_date", "target_date", "ts_code"]
    splits: dict[str, dict[str, list[np.ndarray]]] = {
        name: {f: [] for f in field_names}
        for name in ("train", "val", "test")
    }

    for ts_code, g in df.groupby("ts_code", sort=False):
        g = g.sort_values("trade_date", kind="mergesort").reset_index(drop=True)
        n = int(g.shape[0])
        n_windows = n - seq_len
        if n_windows <= 0:
            continue

        train_end = int(g["is_train"].sum())
        if train_end <= seq_len:
            continue
        val_start = int(train_end * (1.0 - val_ratio_in_train))

        # 每个 stock 只提取一次 numpy 数组
        feat = g["scaled_value"].to_numpy(dtype=np.float32)
        close_r = g["close_raw"].to_numpy(dtype=np.float32)
        min_r = g["min_ref"].to_numpy(dtype=np.float32)
        max_r = g["max_ref"].to_numpy(dtype=np.float32)
        tdate = g["trade_date"].to_numpy(dtype=np.int32)

        # 一次性生成所有滑动窗口 (n_windows, seq_len, 1)
        x = sliding_window_view(feat, seq_len)[:n_windows, :, np.newaxis].copy()
        # 目标和辅助数组：对应窗口 t ∈ [seq_len, n)
        y_s = feat[seq_len:].copy()
        y_r = close_r[seq_len:].copy()
        cur_c = close_r[seq_len - 1 : n - 1].copy()
        next_c = close_r[seq_len:].copy()
        m_min = min_r[seq_len:].copy()
        m_max = max_r[seq_len:].copy()
        sig_d = tdate[seq_len - 1 : n - 1].copy()
        tgt_d = tdate[seq_len:].copy()

        # 分片掩码
        t_idx = np.arange(seq_len, n)
        train_msk = t_idx < val_start
        val_msk = (t_idx >= val_start) & (t_idx < train_end)
        test_msk = t_idx >= train_end

        ts_code_arr = np.array([ts_code] * n_windows, dtype=object)

        for msk, split_name in [(train_msk, "train"), (val_msk, "val"), (test_msk, "test")]:
            if not msk.any():
                continue
            s = splits[split_name]
            s["x"].append(x[msk])
            s["y_scaled"].append(y_s[msk])
            s["y_raw"].append(y_r[msk])
            s["cur_close"].append(cur_c[msk])
            s["next_close"].append(next_c[msk])
            s["min_ref"].append(m_min[msk])
            s["max_ref"].append(m_max[msk])
            s["signal_date"].append(sig_d[msk])
            s["target_date"].append(tgt_d[msk])
            s["ts_code"].append(ts_code_arr[msk])

    def _concat(arr_list: list[np.ndarray]) -> np.ndarray:
        if not arr_list:
            return np.empty((0,), dtype=np.float32)
        return np.concatenate(arr_list, axis=0)

    def _build_wa(name: str) -> WindowArrays:
        s = splits[name]
        if not s["x"]:
            return _empty_windowarrays(scale_name)
        return WindowArrays(
            scale_name=scale_name,
            x=np.concatenate(s["x"], axis=0),
            y_scaled=_concat(s["y_scaled"]),
            y_raw=_concat(s["y_raw"]),
            cur_close=_concat(s["cur_close"]),
            next_close=_concat(s["next_close"]),
            min_ref=_concat(s["min_ref"]),
            max_ref=_concat(s["max_ref"]),
            signal_date=np.concatenate(s["signal_date"], axis=0).astype(np.int32, copy=False),
            target_date=np.concatenate(s["target_date"], axis=0).astype(np.int32, copy=False),
            ts_code=np.concatenate(s["ts_code"], axis=0),
        )

    return (_build_wa("train"), _build_wa("val"), _build_wa("test"))


def build_recent_windows(df: pd.DataFrame, scale_name: str, seq_len: int, last_n_days: int) -> WindowArrays:
    """为最近 N 个信号日生成滚动预测窗口。"""

    field_names = ["x", "y_scaled", "y_raw", "cur_close", "next_close",
                   "min_ref", "max_ref", "signal_date", "target_date", "ts_code"]
    buckets: dict[str, list[np.ndarray]] = {f: [] for f in field_names}

    for ts_code, g in df.groupby("ts_code", sort=False):
        g = g.sort_values("trade_date", kind="mergesort").reset_index(drop=True)
        n = int(g.shape[0])
        n_windows = n - seq_len
        if n_windows <= 0:
            continue

        feat = g["scaled_value"].to_numpy(dtype=np.float32)
        close_r = g["close_raw"].to_numpy(dtype=np.float32)
        min_r = g["min_ref"].to_numpy(dtype=np.float32)
        max_r = g["max_ref"].to_numpy(dtype=np.float32)
        tdate = g["trade_date"].to_numpy(dtype=np.int32)

        x_all = sliding_window_view(feat, seq_len)[:n_windows, :, np.newaxis].copy()

        end_idx = np.arange(seq_len - 1, n - 1, dtype=np.int32)
        if last_n_days > 0 and end_idx.size > last_n_days:
            end_idx = end_idx[-last_n_days:]

        for end_i in end_idx:
            wi = int(end_i) - seq_len + 1
            next_date = int(tdate[end_i + 1]) if end_i + 1 < n else 0
            next_close = float(close_r[end_i + 1]) if end_i + 1 < n else np.nan

            buckets["x"].append(x_all[wi])
            buckets["y_scaled"].append(np.float32(np.nan))
            buckets["y_raw"].append(np.float32(np.nan))
            buckets["cur_close"].append(close_r[end_i])
            buckets["next_close"].append(np.float32(next_close))
            buckets["min_ref"].append(min_r[end_i])
            buckets["max_ref"].append(max_r[end_i])
            buckets["signal_date"].append(tdate[end_i])
            buckets["target_date"].append(np.int32(next_date))
            buckets["ts_code"].append(ts_code)

    if not buckets["x"]:
        return _empty_windowarrays(scale_name)

    return WindowArrays(
        scale_name=scale_name,
        x=np.stack(buckets["x"], axis=0).astype(np.float32, copy=False),
        y_scaled=np.array(buckets["y_scaled"], dtype=np.float32),
        y_raw=np.array(buckets["y_raw"], dtype=np.float32),
        cur_close=np.array(buckets["cur_close"], dtype=np.float32),
        next_close=np.array(buckets["next_close"], dtype=np.float32),
        min_ref=np.array(buckets["min_ref"], dtype=np.float32),
        max_ref=np.array(buckets["max_ref"], dtype=np.float32),
        signal_date=np.array(buckets["signal_date"], dtype=np.int32),
        target_date=np.array(buckets["target_date"], dtype=np.int32),
        ts_code=np.array(buckets["ts_code"], dtype=object),
    )


def _empty_windowarrays(scale_name: str) -> WindowArrays:
    empty_x = np.empty((0, 1, 1), dtype=np.float32)
    empty_f = np.empty((0,), dtype=np.float32)
    empty_i = np.empty((0,), dtype=np.int32)
    empty_s = np.empty((0,), dtype=object)
    return WindowArrays(scale_name, empty_x, empty_f, empty_f, empty_f, empty_f, empty_f, empty_f, empty_i, empty_i, empty_s)
