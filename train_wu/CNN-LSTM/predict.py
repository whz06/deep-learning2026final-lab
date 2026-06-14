from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from stock_predictor.config import RESULT_DIR
from stock_predictor.data import build_recent_windows, load_scale_frame
from stock_predictor.strategy import top_picks_report
from stock_predictor.trainer import load_checkpoint, predict_arrays


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="用训练好的模型生成最新建仓建议")
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--last_days", type=int, default=3, help="输出最近几个信号日")
    parser.add_argument("--top_k", type=int, default=10, help="每个信号日输出前几只股票")
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"])
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    model_path = Path(args.model_path)
    model, cfg, device = load_checkpoint(model_path, device_name=args.device)
    df = load_scale_frame(cfg.scale_name)
    recent_arrays = build_recent_windows(df, scale_name=cfg.scale_name, seq_len=cfg.seq_len, last_n_days=args.last_days)
    signals = predict_arrays(model, recent_arrays, batch_size=4096, device=device, scale_name=cfg.scale_name)

    prefix = model_path.stem
    signal_path = RESULT_DIR / f"{prefix}_recent_signals.csv.gz"
    report_path = RESULT_DIR / f"{prefix}_recent_top{args.top_k}.md"
    signals.to_csv(signal_path, index=False, encoding="utf-8")
    top_picks_report(signals, top_k=int(args.top_k), last_n_days=int(args.last_days), out_path=report_path)

    print(signal_path)
    print(report_path)


if __name__ == "__main__":
    main()
