"""Test that the augmented data loads correctly and quickly."""
from __future__ import annotations
import sys
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import time
from stock_predictor.data import load_scale_frame

t0 = time.time()
print("Loading maxmin data...", flush=True)
df = load_scale_frame("maxmin")
t1 = time.time()
print(f"  {len(df)} rows, {df['ts_code'].nunique()} stocks", flush=True)
print(f"  Date range: {df['trade_date'].min()} to {df['trade_date'].max()}", flush=True)
print(f"  Load time: {t1-t0:.1f}s", flush=True)

# Check the last few rows from newest stocks
print(f"\nLatest trading dates:", flush=True)
latest = df.sort_values('trade_date').groupby('ts_code').tail(1)
print(f"  Total stocks with data: {len(latest)}", flush=True)
print(f"  Max trade_date: {latest['trade_date'].max()}", flush=True)
print(f"  Stocks at 20260605: {(latest['trade_date'] == 20260605).sum()}", flush=True)
