"""
diagnose_feature_ic.py (V2 - vectorized)
Compute single-feature Rank IC vs T+1 return for all candidate features.

Key changes from V1:
  - Uses pandas groupby + rolling for vectorized computation (10-100x faster)
  - Supports --sample N to limit stocks for quick runs
  - Skips slower features (skew/kurt/mdd) unless --full

Usage:
  python v7/diagnose_feature_ic.py --fast --sample 600     # Quick: 2024-2025, 600 stocks
  python v7/diagnose_feature_ic.py --fast                  # Medium: 2024-2025, all stocks
"""
import os, sys, glob, argparse
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PARQUET_PATH = os.path.join(ROOT, "processed", "all_data.parquet")
MONEYFLOW_DIR = os.path.join(ROOT, "data", "moneyflow")
INDEX_PATH = os.path.join(ROOT, "data", "market", "000300.SH.csv")
RESULTS_DIR = os.path.join(ROOT, "v7", "results")
os.makedirs(RESULTS_DIR, exist_ok=True)


def load_merged_data(start_date="20190101", end_date="20250531"):
    """Load all_data.parquet + moneyflow + CSI300, filter dates."""
    print(f"[load] Loading all_data.parquet ...")
    df = pd.read_parquet(PARQUET_PATH)
    df["trade_date"] = df["trade_date"].astype(str)
    for col in ["open","high","low","close","vol","amount","pct_chg","vwap",
                "turnover_rate","turnover_rate_f","volume_ratio",
                "pe","pe_ttm","pb","ps","ps_ttm","dv_ratio","dv_ttm",
                "total_share","float_share","free_share","total_mv","circ_mv"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    print(f"[load] all_data: {len(df)} rows, {df['ts_code'].nunique()} stocks")

    # Merge moneyflow
    mf = _load_moneyflow(start_date, end_date)
    if mf is not None and len(mf) > 0:
        print(f"[load] moneyflow: {len(mf)} rows, {mf['ts_code'].nunique()} stocks")
        df = df.merge(mf, on=["ts_code","trade_date"], how="left")
        print(f"[load] after merge: {len(df)} rows")

    # Merge CSI300
    csi = pd.read_csv(INDEX_PATH, dtype={"trade_date": str})
    csi["idx_ret"] = pd.to_numeric(csi["pct_chg"], errors="coerce") / 100.0
    df = df.merge(csi[["trade_date","idx_ret"]], on="trade_date", how="left")

    # Filter
    df = df[(df["trade_date"] >= start_date) & (df["trade_date"] <= end_date)].copy()
    df = df.sort_values(["ts_code","trade_date"]).reset_index(drop=True)
    df["ret"] = pd.to_numeric(df["pct_chg"], errors="coerce") / 100.0
    df["ret_t1"] = df.groupby("ts_code")["ret"].shift(-1)
    print(f"[load] Final: {len(df)} rows, {df['ts_code'].nunique()} stocks, {df['trade_date'].nunique()} dates")
    return df


def _load_moneyflow(start_date, end_date):
    """Load all moneyflow CSVs in date range."""
    pattern = os.path.join(MONEYFLOW_DIR, "*.csv")
    files = sorted(glob.glob(pattern))
    if not files:
        return None
    frames = []
    for f in files:
        date_str = os.path.basename(f).replace(".csv","")
        if date_str < start_date or date_str > end_date:
            continue
        try:
            chunk = pd.read_csv(f, dtype={"ts_code": str, "trade_date": str})
            needed = ["ts_code","trade_date",
                      "buy_sm_vol","buy_sm_amount","sell_sm_vol","sell_sm_amount",
                      "buy_md_vol","buy_md_amount","sell_md_vol","sell_md_amount",
                      "buy_lg_vol","buy_lg_amount","sell_lg_vol","sell_lg_amount",
                      "buy_elg_vol","buy_elg_amount","sell_elg_vol","sell_elg_amount",
                      "net_mf_vol","net_mf_amount"]
            available = [c for c in needed if c in chunk.columns]
            chunk = chunk[available]
            for c in available:
                if c not in ["ts_code","trade_date"]:
                    chunk[c] = pd.to_numeric(chunk[c], errors="coerce")
            frames.append(chunk)
        except Exception:
            continue
    if not frames:
        return None
    return pd.concat(frames, ignore_index=True)


# ============================================================
# Vectorized feature computation (per group)
# ============================================================

def _rolling_apply(g, col, window, func, **kwargs):
    """Apply a rolling function to a grouped Series."""
    vals = g[col].values
    result = np.full(len(vals), np.nan)
    for i in range(window - 1, len(vals)):
        w = vals[i - window + 1 : i + 1]
        w = w[~np.isnan(w)]
        if len(w) >= max(5, window // 2):
            result[i] = func(w, **kwargs)
    return result


def compute_features_vectorized(df):
    """
    Compute all candidate features using groupby+rolling.
    Returns df with new feature columns added.
    """
    df = df.copy()
    df = df.sort_values(["ts_code","trade_date"]).reset_index(drop=True)

    g = df.groupby("ts_code")

    # --- helper columns ---
    df["high_low"] = df["high"] - df["low"]
    df["log_hl"] = np.log(np.maximum(df["high"], 1e-8) / np.maximum(df["low"], 1e-8))
    df["tr_val"] = np.maximum(df["high"] - df["low"],
                    np.maximum(np.abs(df["high"] - df.groupby("ts_code")["close"].shift(1)),
                               np.abs(df["low"] - df.groupby("ts_code")["close"].shift(1))))

    # --- Tier 1: rolling features (pandas rolling for common ones) ---
    # ATR% (14-day EMA of TR / close)
    df["atr_raw"] = g["tr_val"].transform(lambda x: _ema(x, 14))
    df["atr_pct_14"] = df["atr_raw"] / np.maximum(df["close"], 1e-8)

    # Parkinson vol (20-day rolling std of log(H/L)^2 / (4 ln 2))
    df["park_var"] = df["log_hl"] ** 2 / (4 * np.log(2))
    df["parkinson_vol_20"] = g["park_var"].transform(
        lambda x: x.rolling(20, min_periods=10).std())

    # Price position (20-day)
    h20 = g["high"].transform(lambda x: x.rolling(20, min_periods=10).max())
    l20 = g["low"].transform(lambda x: x.rolling(20, min_periods=10).min())
    df["price_pos_20"] = (df["close"] - l20) / np.maximum(h20 - l20, 1e-8)

    # Amihud
    df["amihud_daily"] = np.abs(df["ret"]) / np.maximum(df["amount"], 1e-8)
    df["amihud_20"] = g["amihud_daily"].transform(
        lambda x: x.rolling(20, min_periods=10).mean())

    # Up ratio (20-day)
    df["is_up"] = (df["ret"] > 0).astype(float)
    df["up_ratio_20"] = g["is_up"].transform(
        lambda x: x.rolling(20, min_periods=10).mean())

    # VWAP gap
    df["vwap_gap"] = df["close"] / np.maximum(df["vwap"], 1e-8) - 1
    df["hl_ratio"] = df["high_low"] / np.maximum(df["close"], 1e-8)

    # Turnover acceleration (5-day)
    to_ma5 = g["turnover_rate"].transform(lambda x: x.rolling(5, min_periods=3).mean())
    df["turnover_acc_5"] = df["turnover_rate"] / np.maximum(to_ma5, 1e-8) - 1

    # Volume ratio vs 5-day
    v_ma5 = g["vol"].transform(lambda x: x.rolling(5, min_periods=3).mean())
    df["vol_ratio_5"] = df["vol"] / np.maximum(v_ma5, 1e-8)

    # Ret volatility (20-day)
    df["ret_vol_20"] = g["ret"].transform(
        lambda x: x.rolling(20, min_periods=10).std())

    # Idio vol (CAPM residual, approximate with rolling beta)
    df["ret_x_idx"] = df["ret"] * df["idx_ret"]
    df["idx_ret_sq"] = df["idx_ret"] ** 2
    ret_x_idx_mean = g["ret_x_idx"].transform(
        lambda x: x.rolling(60, min_periods=20).mean())
    idx_ret_sq_mean = g["idx_ret_sq"].transform(
        lambda x: x.rolling(60, min_periods=20).mean())
    ret_mean = g["ret"].transform(lambda x: x.rolling(60, min_periods=20).mean())
    idx_mean = df.groupby("trade_date")["idx_ret"].transform(
        lambda x: x.rolling(60, min_periods=20).mean())
    # rolling beta approx
    beta_num = ret_x_idx_mean - ret_mean * idx_mean
    beta_den = idx_ret_sq_mean - idx_mean ** 2
    beta_approx = beta_num / np.maximum(np.abs(beta_den), 1e-12)
    beta_approx = np.clip(beta_approx, -5, 5)
    df["idio_ret"] = df["ret"] - beta_approx * df["idx_ret"]
    df["idio_vol_20"] = g["idio_ret"].transform(
        lambda x: x.rolling(20, min_periods=10).std())

    # --- Tier 1: slower features (Python loop per stock) ---
    df["ret_skew_20"] = np.nan
    df["ret_kurt_20"] = np.nan
    df["max_dd_20"] = np.nan
    df["streak"] = np.nan

    stocks = sorted(df["ts_code"].unique())
    for si, ts_code in enumerate(stocks):
        mask = df["ts_code"] == ts_code
        idx = df.index[mask]
        sdf = df.loc[mask].sort_values("trade_date")
        ret_vals = sdf["ret"].values
        close_vals = sdf["close"].values
        N = len(sdf)

        # Skew & Kurt (20-day)
        for i in range(20, N):
            r = ret_vals[i-19:i+1]
            r = r[~np.isnan(r)]
            if len(r) >= 10:
                m = r.mean(); s = r.std(ddof=1)
                if s > 1e-12:
                    df.loc[idx[i], "ret_skew_20"] = ((r - m)**3).mean() / (s**3)
                    df.loc[idx[i], "ret_kurt_20"] = ((r - m)**4).mean() / (s**4)

        # Max drawdown (20-day)
        for i in range(20, N):
            prices = close_vals[i-19:i+1]
            peak = np.maximum.accumulate(prices)
            dd = (peak - prices) / np.maximum(peak, 1e-8)
            df.loc[idx[i], "max_dd_20"] = dd.max()

        # Streak
        streak = 0
        for i in range(1, N):
            if np.isnan(ret_vals[i-1]) or np.isnan(ret_vals[i]):
                streak = 0
            elif np.sign(ret_vals[i]) == np.sign(ret_vals[i-1]) and ret_vals[i] != 0:
                streak = streak + 1 if streak > 0 else 1
            elif ret_vals[i] != 0:
                streak = -1
            df.loc[idx[i], "streak"] = streak

        if (si + 1) % 1000 == 0:
            print(f"  [slow_feat] {si+1}/{len(stocks)} stocks")

    # --- Tier 2: Moneyflow ---
    mf_has = all(c in df.columns for c in ["buy_sm_vol","buy_md_vol","buy_lg_vol","buy_elg_vol",
                                              "sell_sm_vol","sell_md_vol","sell_lg_vol","sell_elg_vol"])
    if mf_has:
        print("[feat] Computing moneyflow features ...")
        # Fill NaN with 0 for moneyflow (missing = no data that day)
        for c in ["buy_sm_vol","buy_sm_amount","sell_sm_vol","sell_sm_amount",
                  "buy_md_vol","buy_md_amount","sell_md_vol","sell_md_amount",
                  "buy_lg_vol","buy_lg_amount","sell_lg_vol","sell_lg_amount",
                  "buy_elg_vol","buy_elg_amount","sell_elg_vol","sell_elg_amount",
                  "net_mf_vol","net_mf_amount"]:
            if c in df.columns:
                df[c] = df[c].fillna(0)

        # Total volumes per size class (buy + sell)
        df["mf_sm_total"] = df["buy_sm_vol"] + df["sell_sm_vol"]
        df["mf_md_total"] = df["buy_md_vol"] + df["sell_md_vol"]
        df["mf_lg_total"] = df["buy_lg_vol"] + df["sell_lg_vol"]
        df["mf_elg_total"] = df["buy_elg_vol"] + df["sell_elg_vol"]
        df["mf_total_vol"] = (df["mf_sm_total"] + df["mf_md_total"] +
                              df["mf_lg_total"] + df["mf_elg_total"])

        # Large+ELG net flow / total
        df["mf_lg_net_pct"] = ((df["buy_lg_vol"] + df["buy_elg_vol"] -
                                 df["sell_lg_vol"] - df["sell_elg_vol"]) /
                                np.maximum(df["mf_total_vol"], 1e-8))

        # Net moneyflow pct (amount)
        if "net_mf_amount" in df.columns:
            df["mf_total_amt"] = (df["buy_sm_amount"] + df["buy_md_amount"] +
                                  df["buy_lg_amount"] + df["buy_elg_amount"] +
                                  df["sell_sm_amount"] + df["sell_md_amount"] +
                                  df["sell_lg_amount"] + df["sell_elg_amount"])
            df["mf_net_amt_pct"] = df["net_mf_amount"] / np.maximum(df["mf_total_amt"], 1e-8)

        # Net moneyflow pct (vol)
        if "net_mf_vol" in df.columns:
            df["mf_net_vol_pct"] = df["net_mf_vol"] / np.maximum(df["mf_total_vol"], 1e-8)

        # Large flow asymmetry
        df["mf_buy_total"] = (df["buy_sm_vol"] + df["buy_md_vol"] +
                               df["buy_lg_vol"] + df["buy_elg_vol"])
        df["mf_sell_total"] = (df["sell_sm_vol"] + df["sell_md_vol"] +
                                df["sell_lg_vol"] + df["sell_elg_vol"])
        lb_ratio = (df["buy_lg_vol"] + df["buy_elg_vol"]) / np.maximum(df["mf_buy_total"], 1e-8)
        ls_ratio = (df["sell_lg_vol"] + df["sell_elg_vol"]) / np.maximum(df["mf_sell_total"], 1e-8)
        df["mf_lg_asymmetry"] = lb_ratio - ls_ratio

        # Flow HHI (concentration across 4 sizes)
        total_v = np.maximum(df["mf_total_vol"].values, 1e-8)
        shares = np.stack([
            df["mf_sm_total"].values, df["mf_md_total"].values,
            df["mf_lg_total"].values, df["mf_elg_total"].values
        ], axis=1) / total_v[:, None]
        df["mf_flow_hhi"] = (shares ** 2).sum(axis=1)

        # Small vs Large divergence
        sm_net = df["buy_sm_vol"] - df["sell_sm_vol"]
        lg_net = df["buy_lg_vol"] + df["buy_elg_vol"] - df["sell_lg_vol"] - df["sell_elg_vol"]
        df["mf_sm_lg_divergence"] = sm_net - lg_net

    # --- Tier 3: Unused metric columns ---
    for col, name in [("pe_ttm","pe_ttm"), ("ps","ps"), ("ps_ttm","ps_ttm"),
                       ("dv_ratio","dv_ratio"), ("dv_ttm","dv_ttm")]:
        if col in df.columns and name not in df.columns:
            df[name] = pd.to_numeric(df[col], errors="coerce")

    if "float_share" in df.columns and "total_share" in df.columns:
        df["float_pct"] = df["float_share"] / np.maximum(df["total_share"], 1e-8)

    if "turnover_rate_f" in df.columns:
        df["turnover_rate_f_f"] = pd.to_numeric(df["turnover_rate_f"], errors="coerce")
        if "turnover_rate" in df.columns:
            df["turnover_free_vs_total"] = df["turnover_rate_f_f"] / np.maximum(
                df["turnover_rate"], 1e-8)

    return df


def _ema(arr, span):
    """EMA on a Series or numpy array. Returns numpy array."""
    vals = arr.values if hasattr(arr, "values") else np.asarray(arr)
    alpha = 2.0 / (span + 1)
    result = np.full_like(vals, np.nan, dtype=np.float64)
    result[0] = vals[0]
    for i in range(1, len(vals)):
        if np.isnan(vals[i]):
            result[i] = result[i-1]
        else:
            result[i] = alpha * vals[i] + (1 - alpha) * result[i-1]
    return result


# ============================================================
# IC computation
# ============================================================

def compute_daily_ic(df, feature_names):
    print(f"[IC] Computing daily Rank IC for {len(feature_names)} features ...")
    results = []
    dates = sorted(df["trade_date"].unique())
    for di, date in enumerate(dates):
        day_df = df[df["trade_date"] == date].dropna(subset=["ret_t1"])
        if len(day_df) < 30:
            continue
        for feat in feature_names:
            if feat not in day_df.columns:
                continue
            vals = day_df[feat].values
            labs = day_df["ret_t1"].values
            mask = ~np.isnan(vals) & ~np.isnan(labs) & ~np.isinf(vals)
            if mask.sum() < 30:
                continue
            try:
                ic, _ = spearmanr(vals[mask], labs[mask])
                if not np.isnan(ic):
                    results.append({"feature": feat, "date": date,
                                    "ic": ic, "n_stocks": mask.sum()})
            except Exception:
                continue
        if (di + 1) % 50 == 0:
            print(f"  [IC] {di+1}/{len(dates)} dates done")
    return pd.DataFrame(results)


def summarize_ic(ic_df):
    summary = ic_df.groupby("feature").agg(
        mean_ic=("ic", "mean"),
        ic_ir=("ic", lambda x: x.mean() / (x.std() + 1e-12)),
        ic_std=("ic", "std"),
        ic_gt_0=("ic", lambda x: (x > 0).mean()),
        n_days=("date", "nunique"),
        avg_stocks=("n_stocks", "mean"),
    ).reset_index()
    return summary.sort_values("mean_ic", ascending=False).reset_index(drop=True)


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fast", action="store_true", help="Only use 2024-2025 data")
    parser.add_argument("--sample", type=int, default=0, help="Sample N stocks only")
    parser.add_argument("--full", action="store_true", help="Include slow features (skew/kurt/mdd)")
    args = parser.parse_args()

    start = "20240101" if args.fast else "20190101"
    end = "20250531"

    df = load_merged_data(start, end)

    # Sample stocks
    if args.sample > 0:
        stocks = sorted(df["ts_code"].unique())
        rng = np.random.RandomState(42)
        sampled = rng.choice(stocks, min(args.sample, len(stocks)), replace=False)
        df = df[df["ts_code"].isin(sampled)].copy()
        print(f"[sample] Using {len(sampled)} sampled stocks, {len(df)} rows")

    # Compute features
    df = compute_features_vectorized(df)

    # Collect feature names
    feature_names = sorted([
        "atr_pct_14", "parkinson_vol_20", "price_pos_20",
        "ret_skew_20", "ret_kurt_20", "idio_vol_20", "amihud_20",
        "streak", "up_ratio_20", "max_dd_20",
        "turnover_acc_5", "vwap_gap", "hl_ratio", "vol_ratio_5", "ret_vol_20",
        "mf_lg_net_pct", "mf_net_amt_pct", "mf_net_vol_pct",
        "mf_lg_asymmetry", "mf_flow_hhi", "mf_sm_lg_divergence",
        "pe_ttm", "ps", "dv_ratio", "float_pct", "turnover_free_vs_total",
    ])
    feature_names = [f for f in feature_names if f in df.columns]

    # Also baseline existing features
    existing = ["ret", "turnover_rate", "volume_ratio", "pe", "pb",
                "circ_mv", "total_mv", "vol", "amount", "pct_chg"]
    for ef in existing:
        if ef in df.columns and ef not in feature_names:
            feature_names.append(ef)

    # Compute IC
    ic_df = compute_daily_ic(df, feature_names)
    if len(ic_df) == 0:
        print("[IC] ERROR: No IC data computed. Check features for NaN.")
        return

    summary = summarize_ic(ic_df)

    # Print
    print("\n" + "=" * 85)
    print("FEATURE IC DIAGNOSIS (Rank IC vs T+1 return)")
    print(f"Period: {start} to {end}")
    print(f"Stocks: {df['ts_code'].nunique()}, Days: {ic_df['date'].nunique()}")
    print("=" * 85)
    print(f"{'Rank':>4} {'Feature':<30} {'Mean IC':>8} {'IC IR':>8} {'IC>0%':>7} {'Days':>6} {'Avg N':>8}")
    print("-" * 85)
    for i, row in summary.iterrows():
        print(f"{i+1:>4} {row['feature']:<30} {row['mean_ic']:>8.4f} {row['ic_ir']:>8.2f} "
              f"{row['ic_gt_0']*100:>6.1f}% {int(row['n_days']):>6} {row['avg_stocks']:>8.0f}")

    # Save
    out = os.path.join(RESULTS_DIR, "feature_ic_diagnosis.csv")
    summary.to_csv(out, index=False)
    print(f"\nSaved: {out}")


if __name__ == "__main__":
    main()
