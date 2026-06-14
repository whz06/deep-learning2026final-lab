"""
Append sim_data/20260608.csv rows to maxmin_scale_daily_close.csv
with proper running min/max scaling, then run predict.
"""
from __future__ import annotations
import sys, subprocess
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import numpy as np
import pandas as pd
from stock_predictor.config import MAXMIN_CSV

SIM_JUN8 = PROJECT_ROOT / "sim_data" / "20260608.csv"

# 1. Load existing CSV
print("Loading existing maxmin CSV...", flush=True)
existing = pd.read_csv(MAXMIN_CSV)
existing["trade_date"] = existing["trade_date"].astype(str)
print(f"  Existing rows: {len(existing)}, last dates: {sorted(existing['trade_date'].unique())[-5:]}")

# 2. Get latest state per stock (running min/max)
print("Getting latest state per stock...", flush=True)
last_state = {}
for ts_code, g in existing.groupby("ts_code"):
    g = g.sort_values("trade_date", kind="mergesort")
    last_row = g.iloc[-1]
    last_state[ts_code] = {
        "min_ref": float(last_row["min_ref"]),
        "max_ref": float(last_row["max_ref"]),
    }

# 3. Load sim_data June 8
print(f"Loading sim_data: {SIM_JUN8.name}", flush=True)
sim8 = pd.read_csv(SIM_JUN8)
sim8["ts_code"] = sim8["ts_code"].astype(str).str.upper().str.strip()
print(f"  Rows: {len(sim8)}, stocks: {sim8['ts_code'].nunique()}")

# 4. Compute scaled values and build new rows
new_rows = []
missing_stocks = []
for _, row in sim8.iterrows():
    ts_code = str(row["ts_code"]).upper().strip()
    trade_date = str(row["trade_date"])
    close = float(row["close"])

    if ts_code in last_state:
        prev_min = last_state[ts_code]["min_ref"]
        prev_max = last_state[ts_code]["max_ref"]
    else:
        # Stock not in existing data — use close as both min and max
        prev_min = close
        prev_max = close
        missing_stocks.append(ts_code)

    close_noise = close
    new_min = min(prev_min, close_noise)
    new_max = max(prev_max, close_noise)

    if new_max > new_min:
        close_scaled = (close_noise - new_min) / (new_max - new_min)
    else:
        close_scaled = 0.0

    new_rows.append({
        "ts_code": ts_code,
        "trade_date": trade_date,
        "close": close,
        "close_raw": close,
        "close_noise": close_noise,
        "min_ref": new_min,
        "max_ref": new_max,
        "close_scaled": close_scaled,
        "train": 0,
    })

new_df = pd.DataFrame(new_rows)
combined = pd.concat([existing, new_df], ignore_index=True)
combined.to_csv(MAXMIN_CSV, index=False)

print(f"\nAppended {len(new_df)} rows (Jun 8) to {MAXMIN_CSV.name}")
print(f"  Combined rows: {len(combined)}")
print(f"  Latest date: {combined['trade_date'].astype(str).max()}")
if missing_stocks:
    print(f"  New stocks (not in existing): {len(missing_stocks)}")
print("Done!")
