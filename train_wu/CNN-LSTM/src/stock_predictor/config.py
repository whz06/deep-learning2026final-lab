from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
RESULT_DIR = PROJECT_ROOT / "result"

MAXMIN_CSV = PROJECT_ROOT / "maxmin_scale_daily_close.csv"
SIGMOID_CSV = PROJECT_ROOT / "sigmoid_scale_daily_close.csv"

DEFAULT_SEQ_LENS = [5, 10, 20]
TRAIN_RATIO = 0.8
VAL_RATIO_IN_TRAIN = 0.1

BATCH_SIZE = 256
NUM_WORKERS = 0
LR = 1e-3
MAX_EPOCHS = 50
PATIENCE = 5
DROPOUT = 0.2

LSTM_HIDDEN = 256
CNN_FILTERS = 64

INITIAL_CASH = 1_000_000.0
HOLD_N = 10
ROTATE_K = 3
LOT_SIZE = 100


@dataclass(frozen=True)
class TrainConfig:
    model_name: str
    scale_name: str
    seq_len: int
    batch_size: int = BATCH_SIZE
    num_workers: int = NUM_WORKERS
    lr: float = LR
    max_epochs: int = MAX_EPOCHS
    patience: int = PATIENCE
    dropout: float = DROPOUT
    lstm_hidden: int = LSTM_HIDDEN
    cnn_filters: int = CNN_FILTERS
