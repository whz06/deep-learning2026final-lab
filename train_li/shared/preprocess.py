"""
Step 1: Merge daily + metric CSVs into a single Parquet file.

Input:
  data/daily/*.csv, data/metric/*.csv, data/basic.csv
Output:
  ../processed/all_data.parquet
"""
import os
import sys
import glob
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(ROOT, "data")
PROCESSED_DIR = os.path.join(ROOT, "processed")
OUTPUT_PATH = os.path.join(PROCESSED_DIR, "all_data.parquet")

FEATURE_COLS_DAILY = ["open", "high", "low", "close", "vol", "amount", "pct_chg", "vwap"]
FEATURE_COLS_METRIC = ["turnover_rate", "volume_ratio", "total_mv", "pe", "pb", "circ_mv"]
MERGE_COLS = ["ts_code", "trade_date"]


def load_basic_filter():
    """Load basic.csv and return allowed ts_codes (non-BJ, non-ST)."""
    basic_path = os.path.join(DATA_DIR, "basic.csv")
    df = pd.read_csv(basic_path, dtype={"ts_code": str, "market": str, "name": str})
    df = df[~df["market"].str.contains("北交所", na=False)]
    df = df[~df["name"].str.contains("ST", na=False)]
    allowed = set(df["ts_code"].unique())
    print(f"[preprocess] basic.csv: {len(df)} stocks after filtering (BJ + ST removed)")
    return allowed


def load_csvs_by_date(data_subdir, usecols=None, dtype=None):
    """
    Read all CSV files in a directory, concat into one DataFrame.
    Each file is cross-sectional for one trading day.
    """
    pattern = os.path.join(DATA_DIR, data_subdir, "*.csv")
    files = sorted(glob.glob(pattern))
    print(f"[preprocess] Reading {len(files)} files from {data_subdir}/")

    frames = []
    for i, fpath in enumerate(files):
        try:
            chunk = pd.read_csv(fpath, usecols=usecols, dtype=dtype)
            frames.append(chunk)
        except Exception as e:
            print(f"  Warning: skipping {os.path.basename(fpath)} — {e}")
            continue

        if (i + 1) % 500 == 0:
            print(f"  ... {i + 1}/{len(files)} files read")

    df = pd.concat(frames, ignore_index=True)
    print(f"[preprocess] {data_subdir}: {len(df)} total rows, {df['ts_code'].nunique()} unique stocks")
    return df


def main():
    os.makedirs(PROCESSED_DIR, exist_ok=True)

    # 1. Filter allowed stocks
    allowed_stocks = load_basic_filter()

    # 2. Load daily data
    print("[preprocess] Loading daily data ...")
    daily = load_csvs_by_date("daily", dtype={"ts_code": str, "trade_date": str})

    daily = daily[daily["ts_code"].isin(allowed_stocks)]
    daily = daily[MERGE_COLS + FEATURE_COLS_DAILY]
    for col in FEATURE_COLS_DAILY:
        daily[col] = pd.to_numeric(daily[col], errors="coerce")
    daily = daily.drop_duplicates(subset=MERGE_COLS)
    print(f"[preprocess] daily after filter: {len(daily)} rows")

    # 3. Load metric data
    print("[preprocess] Loading metric data ...")
    metric_cols_read = MERGE_COLS + FEATURE_COLS_METRIC
    metric = load_csvs_by_date("metric", usecols=metric_cols_read,
                               dtype={"ts_code": str, "trade_date": str})

    metric = metric[metric["ts_code"].isin(allowed_stocks)]
    for col in FEATURE_COLS_METRIC:
        metric[col] = pd.to_numeric(metric[col], errors="coerce")
    metric = metric.drop_duplicates(subset=MERGE_COLS)
    print(f"[preprocess] metric after filter: {len(metric)} rows")

    # 4. Merge daily + metric
    print("[preprocess] Merging daily + metric ...")
    merged = daily.merge(metric, on=MERGE_COLS, how="left", suffixes=("", "_m"))
    # Drop duplicate/dangling columns introduced by merge
    for col in list(merged.columns):
        if col.endswith("_m"):
            merged.drop(columns=[col], inplace=True)

    merged = merged.sort_values(["ts_code", "trade_date"]).reset_index(drop=True)

    # 5. Save
    print(f"[preprocess] Saving {len(merged)} rows to {OUTPUT_PATH}")
    merged.to_parquet(OUTPUT_PATH, index=False, compression="snappy")
    print("[preprocess] Done.")


if __name__ == "__main__":
    main()
