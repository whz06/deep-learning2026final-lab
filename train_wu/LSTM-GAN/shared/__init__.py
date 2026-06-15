"""shared — 数据处理与因子提取模块
- preprocess.py: 原始 CSV → Parquet 合并
- build_windows.py: 基础滑动窗口构建（10 维特征）
- build_windows_v7.py: 完整因子提取流水线（26 维，技术指标 + 截面排名，两阶段标准化，T+1 标签）
"""
