"""
Augment maxmin_scale_daily_close.csv and sigmoid_scale_daily_close.csv
with the latest sim_data (2026-06-01 to 2026-06-05).
Then retrain model and test predictions on the last 2 days.
"""
from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import numpy as np
import pandas as pd
from stock_predictor.data import inverse_sigmoid_formula

SIM_DATA_DIR = PROJECT_ROOT / "sim_data"
MAXMIN_CSV = PROJECT_ROOT / "maxmin_scale_daily_close.csv"
SIGMOID_CSV = PROJECT_ROOT / "sigmoid_scale_daily_close.csv"

SIGMOID_START = 0.0
SIGMOID_END = 150.0
SIGMOID_N = abs(SIGMOID_START - SIGMOID_END)
SIGMOID_A = float(np.log(40_000.0))
SIGMOID_B = float(np.log(5e-3))


def sigmoid_forward(raw_price: np.ndarray) -> np.ndarray:
    """Forward sigmoid scaling: price -> scaled_close.

    Formula derived from inverse_sigmoid_formula in data.py:
    score = 2 / (1 + exp(B - (A/N)*(x - START - N)))
    scaled_close = score - 1
    """
    x = np.asarray(raw_price, dtype=np.float64)
    score = 2.0 / (1.0 + np.exp(SIGMOID_B - (SIGMOID_A / SIGMOID_N) * (x - SIGMOID_START - SIGMOID_N)))
    return (score - 1.0).astype(np.float32)


def load_sim_data() -> pd.DataFrame:
    """Load all sim_data files and combine."""
    sim_files = sorted(SIM_DATA_DIR.glob("*.csv"))
    print(f"Found {len(sim_files)} sim_data files: {[f.name for f in sim_files]}")
    frames = []
    for fp in sim_files:
        df = pd.read_csv(fp)
        df["ts_code"] = df["ts_code"].astype(str).str.upper().str.strip()
        frames.append(df)
    combined = pd.concat(frames, ignore_index=True)
    combined = combined.drop_duplicates(subset=["ts_code", "trade_date"])
    print(f"Loaded {len(combined)} rows from sim_data")
    print(f"  Date range: {combined['trade_date'].min()} to {combined['trade_date'].max()}")
    print(f"  Unique stocks: {combined['ts_code'].nunique()}")
    return combined


def augment_maxmin(existing: pd.DataFrame, sim_data: pd.DataFrame) -> pd.DataFrame:
    """Append sim_data to maxmin CSV with running min/max scaling."""
    # Get last state per stock
    last_state = {}
    for ts_code, g in existing.groupby("ts_code"):
        g = g.sort_values("trade_date")
        last_row = g.iloc[-1]
        last_state[ts_code] = {
            "min_ref": float(last_row["min_ref"]),
            "max_ref": float(last_row["max_ref"]),
        }

    print(f"\nExisting maxmin data: {len(existing)} rows, {len(last_state)} stocks")

    new_rows = []
    new_stock_count = 0
    for _, row in sim_data.iterrows():
        ts_code = str(row["ts_code"]).upper().strip()
        trade_date = str(row["trade_date"])
        close = float(row["close"])

        # Get or initialize running min/max
        if ts_code in last_state:
            prev_min = last_state[ts_code]["min_ref"]
            prev_max = last_state[ts_code]["max_ref"]
        else:
            last_state[ts_code] = {"min_ref": close, "max_ref": close}
            prev_min = close
            prev_max = close
            new_stock_count += 1

        # Use close as close_noise (no noise for new data)
        close_noise = close
        new_min = min(prev_min, close_noise)
        new_max = max(prev_max, close_noise)
        last_state[ts_code] = {"min_ref": new_min, "max_ref": new_max}

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
    print(f"Augmented maxmin data: {len(combined)} rows (+{len(new_df)} new, {new_stock_count} new stocks)")
    print(f"  Date range: {combined['trade_date'].min()} to {combined['trade_date'].max()}")
    return combined


def augment_sigmoid(existing: pd.DataFrame, sim_data: pd.DataFrame) -> pd.DataFrame:
    """Append sim_data to sigmoid CSV using sigmoid scaling formula."""
    print(f"\nExisting sigmoid data: {len(existing)} rows")

    new_rows = []
    for _, row in sim_data.iterrows():
        ts_code = str(row["ts_code"]).upper().strip()
        trade_date = str(row["trade_date"])
        close = float(row["close"])
        scaled_close = sigmoid_forward(np.array([close]))[0]
        new_rows.append({
            "ts_code": ts_code,
            "trade_date": trade_date,
            "close": close,
            "scaled_close": scaled_close,
        })

    new_df = pd.DataFrame(new_rows)
    combined = pd.concat([existing, new_df], ignore_index=True)

    # Verify inverse works
    test_close = combined["close"].to_numpy(dtype=np.float32)
    test_scaled = combined["scaled_close"].to_numpy(dtype=np.float32)
    recovered = inverse_sigmoid_formula(test_scaled)
    mae = float(np.mean(np.abs(recovered - test_close)))
    print(f"Augmented sigmoid data: {len(combined)} rows (+{len(new_df)} new)")
    print(f"  Inverse sigmoid MAE: {mae:.6f} (should be < 0.01)")
    return combined


def main():
    print("=" * 60)
    print("Augmenting training data with latest sim_data (2026-06-01 to 2026-06-05)")
    print("=" * 60)

    # 1. Load sim_data
    sim_data = load_sim_data()

    # 2. Augment maxmin
    print("\n" + "-" * 40)
    print("Augmenting maxmin_scale_daily_close.csv")
    print("-" * 40)
    existing_maxmin = pd.read_csv(MAXMIN_CSV)
    # Convert trade_date to string format for consistency
    existing_maxmin["trade_date"] = existing_maxmin["trade_date"].astype(str)
    augmented_maxmin = augment_maxmin(existing_maxmin, sim_data)
    augmented_maxmin.to_csv(MAXMIN_CSV, index=False)
    print(f"  Saved to {MAXMIN_CSV}")

    # 3. Augment sigmoid
    print("\n" + "-" * 40)
    print("Augmenting sigmoid_scale_daily_close.csv")
    print("-" * 40)
    existing_sigmoid = pd.read_csv(SIGMOID_CSV)
    existing_sigmoid["trade_date"] = existing_sigmoid["trade_date"].astype(str)
    augmented_sigmoid = augment_sigmoid(existing_sigmoid, sim_data)
    augmented_sigmoid.to_csv(SIGMOID_CSV, index=False)
    print(f"  Saved to {SIGMOID_CSV}")

    print("\n" + "=" * 60)
    print("Done! Training data is now up to date.")
    print("=" * 60)


if __name__ == "__main__":
    main()
