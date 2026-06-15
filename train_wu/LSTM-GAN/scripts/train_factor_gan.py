from __future__ import annotations

"""
Factor-GAN 训练脚本入口（日频版本）。

直接读取 all_daily_OHLCV_with_indicators.csv 进行 T=5 日滑动窗口训练。
"""

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from factor_gan_lab.data import (
    PREBUILT_N_FACTORS,
    PREBUILT_TIMESTEP,
    parse_factor_columns_arg,
    prepare_training_frame,
)
from factor_gan_lab.models import FactorGANConfig
from factor_gan_lab.training import pick_device, run_rolling_training


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data",
        type=str,
        default=str(PROJECT_ROOT.parent / "data process" / "all_daily_OHLCV_with_indicators.csv"),
        help="日频样本表路径（默认同级目录 data process/all_daily_OHLCV_with_indicators.csv）。",
    )
    parser.add_argument(
        "--factor_cols",
        type=str,
        default="",
        help="逗号分隔的因子列名；留空则自动推断所有数值型非保留列。",
    )
    parser.add_argument(
        "--result_dir",
        type=str,
        default=str(PROJECT_ROOT / "outputs" / "experiments"),
        help="训练结果输出目录。",
    )
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--train_days", type=int, default=504, help="训练窗口交易日数（默认 504 ≈ 2 年）。")
    parser.add_argument("--val_days", type=int, default=63, help="验证窗口交易日数（默认 63 ≈ 3 个月）。")
    parser.add_argument("--test_days", type=int, default=21, help="测试窗口交易日数（默认 21 ≈ 1 个月）。")
    parser.add_argument("--step_days", type=int, default=21, help="窗口滑动步长（默认 21 交易日）。")
    parser.add_argument("--max_epochs", type=int, default=200)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--n_critic", type=int, default=10)
    parser.add_argument("--mse_weight", type=float, default=10.0, help="G_loss 中 MSE 项的权重系数（默认 10.0）。")
    parser.add_argument("--max_windows", type=int, default=0, help="0 表示跑完所有窗口。")
    parser.add_argument(
        "--data_mode", type=str, default="csv", choices=["csv", "prebuilt"],
        help="数据模式: csv (从原始 CSV 实时构建序列, 默认) 或 prebuilt (读取 v7 预构建 .pt 文件)。",
    )
    parser.add_argument(
        "--windows_dir", type=str, default="",
        help="prebuilt 模式下的 .pt 文件目录路径 (如 processed/windows_v7)。",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    device = pick_device(args.device)

    if args.data_mode == "prebuilt":
        # ---- v7 预构建 .pt 数据模式 ----
        windows_dir = str(args.windows_dir) if args.windows_dir else str(PROJECT_ROOT / "processed" / "windows_v7")
        factor_columns = [f"v7_factor_{i:02d}" for i in range(PREBUILT_N_FACTORS)]
        # 用全量日期构建 timeline (prebuilt 不需要 frame, 但 run_rolling_training 用 dates 做切分)
        # 扫描 windows_dir 下的日期来构建 timeline
        from pathlib import Path
        pt_files = sorted(Path(windows_dir).rglob("*.pt"))
        if not pt_files:
            raise FileNotFoundError(f"prebuilt 目录 {windows_dir} 中未找到 .pt 文件。请先运行 build_windows_v7.py。")
        # 从文件名解析日期 (train/2019/20190102.pt -> 2019-01-02)
        import pandas as pd
        dates_set: set[pd.Timestamp] = set()
        for pt_path in pt_files:
            date_str = pt_path.stem  # 如 20190102
            dates_set.add(pd.Timestamp(date_str))
        dates = sorted(dates_set)
        frame = pd.DataFrame({"trade_date": dates})  # 占位, 只用于 timeline
        print(f"prebuilt 模式: {len(pt_files)} 个 .pt 文件, {len(dates)} 个交易日")
        print(f"因子维度: {PREBUILT_N_FACTORS}, 时间步: {PREBUILT_TIMESTEP}")

        run_rolling_training(
            frame=frame,
            factor_columns=factor_columns,
            result_dir=args.result_dir,
            device=device,
            train_days=int(args.train_days),
            val_days=int(args.val_days),
            test_days=int(args.test_days),
            step_days=int(args.step_days),
            max_epochs=int(args.max_epochs),
            patience=int(args.patience),
            batch_size=int(args.batch_size),
            lr=float(args.lr),
            n_critic=int(args.n_critic),
            mse_weight=float(args.mse_weight),
            max_windows=int(args.max_windows),
            windows_dir=windows_dir,
        )
    else:
        # ---- CSV 数据模式 (原有逻辑) ----
        factor_columns = parse_factor_columns_arg(args.factor_cols)
        prepared = prepare_training_frame(args.data, factor_columns=factor_columns)
        print(f"数据加载完成：{len(prepared.frame):,} 条有效样本，{len(prepared.factor_columns)} 个因子")
        print(f"因子列：{prepared.factor_columns}")

        run_rolling_training(
            frame=prepared.frame,
            factor_columns=prepared.factor_columns,
            result_dir=args.result_dir,
            device=device,
            train_days=int(args.train_days),
            val_days=int(args.val_days),
            test_days=int(args.test_days),
            step_days=int(args.step_days),
            max_epochs=int(args.max_epochs),
            patience=int(args.patience),
            batch_size=int(args.batch_size),
            lr=float(args.lr),
            n_critic=int(args.n_critic),
            mse_weight=float(args.mse_weight),
            max_windows=int(args.max_windows),
        )


if __name__ == "__main__":
    main()
