"""
数据层工具。

注意：
- 这里只保留“读取与整理你已经准备好的日频样本表”的通用逻辑
- 不负责任何旧版因子计算、特征构建或原始数据聚合
"""

from .dataset import (
    FactorGanTensors,
    PreparedTabularData,
    PREBUILT_N_FACTORS,
    PREBUILT_TIMESTEP,
    add_next_day_return,
    build_sequences,
    infer_factor_columns,
    load_prebuilt_windows,
    load_processed_frame,
    parse_factor_columns_arg,
    prepare_training_frame,
    rank_to_unit_interval,
)

__all__ = [
    "FactorGanTensors",
    "PreparedTabularData",
    "PREBUILT_N_FACTORS",
    "PREBUILT_TIMESTEP",
    "add_next_day_return",
    "build_sequences",
    "infer_factor_columns",
    "load_prebuilt_windows",
    "load_processed_frame",
    "parse_factor_columns_arg",
    "prepare_training_frame",
    "rank_to_unit_interval",
]
