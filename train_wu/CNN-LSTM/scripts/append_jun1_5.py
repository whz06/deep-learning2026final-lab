"""Append sim_data Jun 1-5 to maxmin CSV (Jun 8 already appended)."""
from __future__ import annotations, print_function
import sys
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import numpy as np
import pandas as pd
from stock_predictor.config import MAXMIN_CSV

SIM_DATA_DIR = PROJECT_ROOT / "sim_data"
SIM_FILES = sorted(SIM_DATA_DIR.glob("*.csv"))

print("Loading existing maxmin CSV...", flush=True)
existing = pd.read_csv(MAXMIN_CSV)
existing["td_str"] = existing["trade_date"].astype(str)

# Keep only original data (dash format) + Jun 8 (already appended)
# Remove any old Jun 1-5 that might exist
is_jun_new = existing["td_str"].str.match(r"^\d{8}$") & (existing["td_str"] >= "20260601") & (existing["td_str"] < "20260608")
if is_jun_new.any():
    print(f"  Removing {is_jun_new.sum()} stale jun 1-5 rows", flush=True)
    existing = existing[~is_jun_new].copy()

existing.drop(columns=["td_str"], inplace=True)
print(f"  Cleaned rows: {len(existing)}", flush=True)
print(f"  Date range: {existing['trade_date'].astype(str).min()} ~ {existing['trade_date'].astype(str).max()}", flush=True)

# Latest min/max per stock
print("Getting latest state per stock...", flush=True)
last_state = {}
for ts_code, g in existing.groupby("ts_code"):
    g = g.sort_values("trade_date", kind="mergesort")
    last_row = g.iloc[-1]
    last_state[ts_code] = {"min_ref": float(last_row["min_ref"]), "max_ref": float(last_row["max_ref"])}

# Load sim_data (Jun 1-5 only, skip Jun 8 since already appended)
print(f"Loading sim_data files: {[f.name for f in SIM_FILES]}", flush=True)
sim_all = []
for fp in SIM_FILES:
    df = pd.read_csv(fp)
    df["ts_code"] = df["ts_code"].astype(str).str.upper().str.strip()
    sim_all.append(df)
sim_all = pd.concat(sim_all, ignore_index=True).drop_duplicates(subset=["ts_code", "trade_date"])

# Filter to only Jun 1-5 (Jun 8 already in CSV)
sim_all = sim_all[sim_all["trade_date"].astype(str) < "20260608"].copy()
print(f"  Sim rows to add (Jun 1-5): {len(sim_all)}", flush=True)

# Compute scaled values
new_rows = []
new_stock_count = 0
for _, row in sim_all.iterrows():
    ts_code = str(row["ts_code"]).upper().strip()
    trade_date = str(row["trade_date"])
    close = float(row["close"])
    if ts_code in last_state:
        pmin, pmax = last_state[ts_code]["min_ref"], last_state[ts_code]["max_ref"]
    else:
        pmin = pmax = close
        new_stock_count += 1
    new_min = min(pmin, close)
    new_max = max(pmax, close)
    last_state[ts_code] = {"min_ref": new_min, "max_ref": new_max}
    scaled = (close - new_min) / (new_max - new_min) if new_max > new_min else 0.0
    new_rows.append({"ts_code": ts_code, "trade_date": trade_date,
                     "close": close, "close_raw": close, "close_noise": close,
                     "min_ref": new_min, "max_ref": new_max, "close_scaled": scaled, "train": 0})

new_df = pd.DataFrame(new_rows)
combined = pd.concat([existing, new_df], ignore_index=True)
combined.to_csv(MAXMIN_CSV, index=False)

print(f"\nAppended {len(new_df)} rows (Jun 1-5)", flush=True)
print(f"Total: {len(combined)} rows", flush=True)
print(f"Latest dates: {sorted(combined['trade_date'].astype(str).unique())[-10:]}", flush=True)
print("Done!", flush=True)
