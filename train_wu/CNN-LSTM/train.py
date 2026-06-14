from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from stock_predictor.config import DEFAULT_SEQ_LENS, RESULT_DIR, TrainConfig
from stock_predictor.data import build_splits, load_scale_frame
from stock_predictor.strategy import run_backtest, save_backtest
from stock_predictor.trainer import save_train_result, train_one_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="训练 GN-CNN-LSTM 系列模型")
    parser.add_argument("--model", type=str, default="lstm", choices=["lstm", "cnn", "cnn_lstm"])
    parser.add_argument("--scale", type=str, default="all", choices=["all", "maxmin", "sigmoid"])
    parser.add_argument("--seq_lens", type=str, default="5,10,20")
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--year", type=int, default=None, help="按年份过滤数据，如 2024")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    seq_lens = _parse_seq_lens(args.seq_lens)
    scale_names = ["maxmin", "sigmoid"] if args.scale == "all" else [args.scale]

    for scale_name in scale_names:
        df = load_scale_frame(scale_name, year=args.year)

        for seq_len in seq_lens:
            cfg = TrainConfig(model_name=args.model, scale_name=scale_name, seq_len=seq_len)
            train_arrays, val_arrays, test_arrays = build_splits(df, scale_name=scale_name, seq_len=seq_len)
            result = train_one_model(cfg, train_arrays, val_arrays, test_arrays, device_name=args.device)
            saved = save_train_result(result, cfg, RESULT_DIR)

            daily, holdings, summary = run_backtest(result["signals"])
            bt_saved = save_backtest(f"{cfg.scale_name}_{cfg.model_name}_L{cfg.seq_len}", daily, holdings, summary)

            print(f"\n[{cfg.scale_name}_{cfg.model_name}_L{cfg.seq_len}] 测试集结果")
            print(f"IC/IR   : ic={result['metrics']['ic']:.6f}, ir={result['metrics']['ir']:.6f}")
            print(f"RankIC/IR: rank_ic={result['metrics']['rank_ic']:.6f}, rank_ir={result['metrics']['rank_ir']:.6f}")
            print(f"方向胜率: {result['metrics']['direction_win_rate']:.6f}")
            print(f"回测    : annual_return={summary['annual_return']:.6f}, max_drawdown={summary['max_drawdown']:.6f}, sharpe={summary['sharpe']:.6f}")
            print(f"文件    : {saved['metrics']}")
            print(f"回测    : {bt_saved['summary']}\n")


def _parse_seq_lens(text: str) -> list[int]:
    values = [int(x.strip()) for x in text.split(",") if x.strip()]
    return values if values else DEFAULT_SEQ_LENS


if __name__ == "__main__":
    main()
