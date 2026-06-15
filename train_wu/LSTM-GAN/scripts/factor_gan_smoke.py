from __future__ import annotations

"""
Factor-GAN 的最小可运行验证脚本（日频版本）。

目标：
- 不做正式训练
- 只验证"数据读取/张量构造/模型前向"这条主链路是否可运行
"""

import argparse
import sys
from pathlib import Path

import pandas as pd
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from factor_gan_lab.data import build_sequences, prepare_training_frame
from factor_gan_lab.models import FactorGAN, FactorGANConfig


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data",
        type=str,
        default=str(PROJECT_ROOT / "data" / "processed" / "features.csv"),
        help="真实数据路径；如果文件不存在，会跳过真实数据前向测试。",
    )
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--synthetic_factors", type=int, default=8)
    return parser.parse_args(argv)


def run_random_forward(device: torch.device, n_factors: int) -> None:
    """用随机张量验证模型前向维度是否正确。"""
    cfg = FactorGANConfig(n_factors=n_factors)
    model = FactorGAN(cfg).to(device)
    z = torch.randn(16, cfg.timestep, cfg.n_factors, device=device)
    r_hat, d_score = model(z)
    _ = model.D(z, torch.randn_like(r_hat))
    print("random_forward_ok", z.shape, r_hat.shape, d_score.shape)


def run_synthetic_pipeline(device: torch.device, n_factors: int) -> None:
    """用合成日频数据验证完整流水线。"""
    rng = torch.Generator().manual_seed(7)
    factor_columns = [f"factor_{index:02d}" for index in range(1, n_factors + 1)]
    stocks = [f"{index:06d}.SZ" for index in range(1, 31)]
    dates = pd.date_range("2016-01-04", periods=60, freq="B")

    rows: list[dict] = []
    for trade_date in dates:
        x = torch.randn(len(stocks), len(factor_columns), generator=rng)
        close = 10.0 + torch.randn(len(stocks), generator=rng).abs() * 5.0
        for row_index, ts_code in enumerate(stocks):
            row = {
                "ts_code": ts_code,
                "trade_date": trade_date,
                "open": float(close[row_index]) - 0.1,
                "high": float(close[row_index]) + 0.2,
                "low": float(close[row_index]) - 0.2,
                "close": float(close[row_index]),
                "vol": float(torch.rand(1, generator=rng).item() * 1e7),
            }
            row.update(
                {
                    factor_columns[col_index]: float(x[row_index, col_index].item())
                    for col_index in range(len(factor_columns))
                }
            )
            rows.append(row)

    synthetic_path = PROJECT_ROOT / "data" / "interim" / "_smoke_synthetic.csv"
    synthetic_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(synthetic_path, index=False, encoding="utf-8-sig")

    prepared = prepare_training_frame(synthetic_path, factor_columns=factor_columns)
    print(f"prepared: {len(prepared.frame)} rows, {len(prepared.factor_columns)} factors")
    tensors = build_sequences(prepared.frame, prepared.factor_columns, timestep=5)
    print(f"sequences: {tensors.z.shape}, y={tensors.y.shape}")

    cfg = FactorGANConfig(n_factors=len(prepared.factor_columns))
    model = FactorGAN(cfg).to(device)
    z = tensors.z[:64].to(device)
    y = tensors.y[:64].to(device)
    r_hat, d_fake = model(z)
    d_real = model.D(z, y)
    print("synthetic_pipeline_ok", z.shape, r_hat.shape, d_fake.shape, d_real.shape)

    synthetic_path.unlink(missing_ok=True)


def run_real_data_forward(data_path: Path, batch_size: int, device: torch.device) -> None:
    """用真实数据做一次前向验证。"""
    if not data_path.exists():
        print(f"skip_real_data_forward (not found): {data_path}")
        return

    prepared = prepare_training_frame(data_path)
    print(f"prepared: {len(prepared.frame)} rows, {len(prepared.factor_columns)} factors")
    tensors = build_sequences(prepared.frame, prepared.factor_columns, timestep=5)
    if len(tensors.y) <= 0:
        print("skip_real_data_forward (no valid sequences)")
        return
    print(f"sequences: {tensors.z.shape}")

    cfg = FactorGANConfig(n_factors=len(prepared.factor_columns))
    model = FactorGAN(cfg).to(device)
    z = tensors.z[:batch_size].to(device)
    y = tensors.y[:batch_size].to(device)
    r_hat, d_score = model(z)
    d_real = model.D(z, y)
    print("real_data_forward_ok", z.shape, r_hat.shape, d_score.shape, d_real.shape)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    device = torch.device(args.device)
    run_random_forward(device, n_factors=int(args.synthetic_factors))
    run_synthetic_pipeline(device, n_factors=int(args.synthetic_factors))
    run_real_data_forward(Path(args.data), batch_size=int(args.batch), device=device)


if __name__ == "__main__":
    main()
