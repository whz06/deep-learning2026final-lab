"""
v1/config.py — Centralised hyperparameter definitions and sweep spaces.

All tunable knobs live here. Import from model.py / train.py.
"""
from dataclasses import dataclass, field
from typing import List, Dict, Any
import itertools

# ====================================================================
# Structural Parameters —  change infrequently
# ====================================================================

FEATURE_COLS: List[str] = [
    "open", "high", "low", "close",
    "vol", "amount", "pct_chg",
    "turnover_rate", "volume_ratio", "total_mv",
]
LABEL_COL: str = "pct_chg"

TRAIN_START: str = "20190102"
TRAIN_END: str   = "20241231"
VAL_START: str   = "20250102"
VAL_END: str     = "20251231"


# ====================================================================
# Default Hyperparameters —  single-run baseline
# ====================================================================

@dataclass
class DataConfig:
    window_size: int   = 60
    batch_size: int    = 2048        # stocks sampled per trading day

@dataclass
class ModelConfig:
    input_dim: int     = len(FEATURE_COLS)   # 10
    hidden_size: int   = 128
    num_layers: int    = 2
    dropout: float     = 0.3
    bidirectional: bool = False

@dataclass
class TrainConfig:
    lr: float              = 1e-3
    weight_decay: float    = 1e-5
    epochs: int            = 50
    patience: int          = 10           # early-stopping on val RankIC
    scheduler_factor: float = 0.5         # ReduceLROnPlateau
    scheduler_patience: int = 5
    grad_clip: float       = 1.0

@dataclass
class ExpConfig:
    name: str
    data: DataConfig   = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)


# ====================================================================
# Sweep Spaces —  phase 1 (coarse) → phase 2 (fine) → phase 3 (refine)
# ====================================================================

PHASE1_SPACE: Dict[str, list] = {
    "window_size": [20, 40, 60],
    "hidden_size": [64, 128, 256],
}

PHASE2_SPACE: Dict[str, list] = {
    "dropout": [0.1, 0.3, 0.5],
    "lr":      [5e-4, 1e-3, 3e-3],
}

PHASE3_SPACE: Dict[str, list] = {
    "num_layers": [1, 2, 3],
}


def build_sweep_configs(space: Dict[str, list], base: ExpConfig) -> List[ExpConfig]:
    """Cartesian product of sweep space → list of ExpConfig."""
    keys = list(space.keys())
    values = list(space.values())
    configs = []

    for combo in itertools.product(*values):
        params = dict(zip(keys, combo))
        cfg = ExpConfig(
            name="",
            data=DataConfig(**base.data.__dict__),
            model=ModelConfig(**base.model.__dict__),
            train=TrainConfig(**base.train.__dict__),
        )

        for k, v in params.items():
            if k == "window_size":
                cfg.data.window_size = int(v)
            elif k == "batch_size":
                cfg.data.batch_size = int(v)
            elif k == "hidden_size":
                cfg.model.hidden_size = int(v)
            elif k == "num_layers":
                cfg.model.num_layers = int(v)
            elif k == "dropout":
                cfg.model.dropout = float(v)
            elif k == "bidirectional":
                cfg.model.bidirectional = bool(v)
            elif k == "lr":
                cfg.train.lr = float(v)
            elif k == "weight_decay":
                cfg.train.weight_decay = float(v)
            elif k == "patience":
                cfg.train.patience = int(v)
            elif k == "epochs":
                cfg.train.epochs = int(v)

        cfg.name = _make_name(cfg, params)
        configs.append(cfg)

    return configs


def _make_name(cfg: ExpConfig, varied: dict) -> str:
    parts = []
    for k, v in varied.items():
        abbr = {"window_size": "T", "hidden_size": "H", "num_layers": "L",
                "dropout": "D", "lr": "LR", "batch_size": "B"}.get(k, k)
        if isinstance(v, float):
            parts.append(f"{abbr}{v}")
        else:
            parts.append(f"{abbr}{v}")
    return "_".join(parts)


# Convenience
DEFAULT_EXP = ExpConfig(name="baseline")
