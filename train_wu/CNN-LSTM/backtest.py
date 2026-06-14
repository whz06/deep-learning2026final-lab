from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from stock_predictor.strategy import run_backtest, save_backtest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="对测试信号做 N-K 回测")
    parser.add_argument("--signals_csv", type=str, required=True)
    parser.add_argument("--n", type=int, default=10, help="持仓股票数")
    parser.add_argument("--k", type=int, default=3, help="每日最多换仓数")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    signals_path = Path(args.signals_csv)
    signals = pd.read_csv(signals_path)
    daily, holdings, summary = run_backtest(signals, n=int(args.n), k=int(args.k))
    saved = save_backtest(signals_path.stem.replace(".csv", ""), daily, holdings, summary)
    print(saved["summary"])


if __name__ == "__main__":
    main()

