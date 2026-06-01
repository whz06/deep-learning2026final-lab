"""
v2/config.py — Centralised config for V2 multi-model training.
"""
from dataclasses import dataclass, field
from typing import List, Dict
import itertools

# ====================================================================
# Shared Data Config
# ====================================================================
WINDOW_SIZE: int = 60
TRAIN_START: str = "20190102"
TRAIN_END: str   = "20241231"
VAL_START: str   = "20250102"
VAL_END: str     = "20251231"
BATCH_SIZE: int  = 2048
BATCH_SIZE_MLP: int = 4096
INPUT_DIM: int   = 22     # 10 raw + 8 tech + 4 cross

# ====================================================================
# Model-specific defaults
# ====================================================================

@dataclass
class MLPConfig:
    input_dim: int     = INPUT_DIM * WINDOW_SIZE    # 1320
    hidden_dim: int    = 512
    n_layers: int      = 3
    dropout: float     = 0.3
    lr: float          = 1e-3
    weight_decay: float = 1e-5

@dataclass
class GRUConfig:
    input_dim: int     = INPUT_DIM
    hidden_size: int   = 128       # V1 best
    num_layers: int    = 2
    dropout: float     = 0.1       # V1 best
    bidirectional: bool = False
    lr: float          = 5e-4      # V1 best
    weight_decay: float = 1e-5

@dataclass
class TFConfig:
    input_dim: int       = INPUT_DIM
    d_model: int         = 96
    n_heads: int         = 4
    n_temporal_layers: int = 2
    n_spatial_layers: int  = 1
    dropout: float       = 0.1
    lr: float            = 5e-4
    weight_decay: float  = 1e-5

# Shared training
EPOCHS: int    = 50
PATIENCE: int  = 10
GRAD_CLIP: float = 1.0
SCHEDULER_FACTOR: float = 0.5
SCHEDULER_PATIENCE: int = 5

# ====================================================================
# Sweep Spaces
# ====================================================================

MLP_SWEEP: Dict[str, list] = {
    "hidden_dim": [256, 512, 1024],
    "n_layers": [2, 3, 4],
    "dropout": [0.1, 0.3],
    "lr": [5e-4, 1e-3],
}

GRU_SWEEP: Dict[str, list] = {
    "num_layers": [1, 2, 3],
    "dropout": [0.1, 0.2, 0.3],
    "lr": [3e-4, 5e-4, 1e-3],
}

TF_SWEEP_COARSE: Dict[str, list] = {
    "d_model": [64, 96, 128],
    "n_heads": [4, 8],
    "n_temporal_layers": [2, 3],
}

TF_SWEEP_FINE: Dict[str, list] = {
    "n_spatial_layers": [1, 2],
    "dropout": [0.1, 0.2, 0.3],
    "lr": [3e-4, 5e-4, 1e-3],
}

# ====================================================================
# Ensemble
# ====================================================================
ENSEMBLE_SEEDS: List[int] = [42, 123, 456, 789, 1024]
FUSION_WEIGHT_STEP: float = 0.1
