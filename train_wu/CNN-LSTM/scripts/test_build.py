"""Test build_splits time for diagnostics."""
from __future__ import annotations
import sys, time
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from stock_predictor.data import load_scale_frame, build_splits

print("Loading data...", flush=True)
df = load_scale_frame("maxmin")
print(f"Loaded: {len(df)} rows, {df['ts_code'].nunique()} stocks", flush=True)

for seq_len in [10, 20]:
    print(f"\nBuilding splits (seq_len={seq_len})...", flush=True)
    t0 = time.time()
    train, val, test = build_splits(df, scale_name="maxmin", seq_len=seq_len)
    t1 = time.time()
    print(f"  Train: {len(train.x):,} windows", flush=True)
    print(f"  Val:   {len(val.x):,} windows", flush=True)
    print(f"  Test:  {len(test.x):,} windows", flush=True)
    print(f"  Time:  {t1-t0:.1f}s", flush=True)
