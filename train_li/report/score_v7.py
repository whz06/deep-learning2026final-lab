"""Minimal V7 Spatial full-universe scoring using original model imports."""
import os, sys, numpy as np, pandas as pd, torch
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, ROOT)
from v7.models.gru_spatial import GRURankerSpatial

# Reuse feature pipeline from score_models
from score_models import (W, WARMUP, RAW, TECH, LOG_COLS, TDIM_V67,
                          add_tech, winsorize, norm_t, norm_c, rank_pct)

CKPT = os.path.join(ROOT, "v7", "checkpoints", "gru_spatial_v7_d32_K5_H128_L1_D0.2_lr0.0003_N1024.pt")
OUT = os.path.join(ROOT, "v7", "results", "daily_scores_spatial_t1.parquet")
PARQUET = os.path.join(ROOT, "processed", "all_data.parquet")
CSI_P = os.path.join(ROOT, "data", "market", "000300.SH.csv")
START, END = "20260105", "20260529"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load model
state = torch.load(CKPT, map_location="cpu", weights_only=True)
state = {k.replace("_orig_mod.", ""): v for k, v in state.items()}
model = GRURankerSpatial(input_dim=26, hidden_size=128, num_layers=1, dropout=0.2, d_proj=32, K=5)
model.load_state_dict(state, strict=True)
model.to(DEVICE).eval()

# Load data
df = pd.read_parquet(PARQUET); df["trade_date"] = df["trade_date"].astype(str)
all_dates = sorted(df["trade_date"].unique())
wc = max(0, all_dates.index(START) - WARMUP)
df = df[df["trade_date"] >= all_dates[wc]]
csi = pd.read_csv(CSI_P, dtype={"trade_date": str})
csi_map = dict(zip(csi["trade_date"], csi["pct_chg"].astype(np.float32)))

# Build feature store (same as score_models but for 26-dim)
print("[*] Building feature store ...")
df = df.sort_values(["ts_code", "trade_date"]).reset_index(drop=True)
temporal_list, lastval_list, date_pos_list, stock_ids = [], [], [], []
for ts, sdf in df.groupby("ts_code", sort=True):
    sdf = sdf.reset_index(drop=True)
    if len(sdf) < W: continue
    for col in RAW:
        if col in sdf.columns: sdf[col] = sdf[col].ffill()
    sdf = add_tech(sdf)
    sdf["vwap_gap"] = (sdf["close"] / sdf["vwap"] - 1) if "vwap" in sdf.columns else 0.0
    raw_a = sdf[RAW].astype(np.float32).values
    tech_a = sdf[TECH].astype(np.float32).values
    vwap_a = sdf["vwap_gap"].astype(np.float32).values.reshape(-1, 1)
    temp = np.concatenate([raw_a, tech_a, vwap_a], axis=1)
    for ci, col in enumerate(RAW):
        if col in LOG_COLS: temp[:, ci] = np.log1p(np.maximum(temp[:, ci], 0))
    temporal_list.append(temp)
    lv = np.column_stack([
        sdf["pct_chg"].astype(np.float32).values,
        sdf["amount"].astype(np.float32).values,
        sdf["turnover_rate"].astype(np.float32).values if "turnover_rate" in sdf.columns else np.zeros(len(sdf), dtype=np.float32),
        sdf["pe"].astype(np.float32).values if "pe" in sdf.columns else np.zeros(len(sdf), dtype=np.float32),
        sdf["pb"].astype(np.float32).values if "pb" in sdf.columns else np.zeros(len(sdf), dtype=np.float32),
        sdf["circ_mv"].astype(np.float32).values if "circ_mv" in sdf.columns else np.zeros(len(sdf), dtype=np.float32),
    ])
    lastval_list.append(lv)
    date_pos_list.append({d: i for i, d in enumerate(sdf["trade_date"].tolist())})
    stock_ids.append(ts)
n_stocks = len(stock_ids)
print(f"[*] {n_stocks} stocks")

# Score dates
target_dates = [d for d in all_dates if START <= d <= END]
existing = pd.read_parquet(OUT) if os.path.exists(OUT) else None
if existing is not None:
    ed = set(existing["trade_date"].unique())
    target_dates = [d for d in target_dates if d not in ed]
    print(f"[*] existing={len(ed)} new={len(target_dates)}")
else:
    print(f"[*] target={len(target_dates)} dates")

all_rows = []
for di, date in enumerate(target_dates):
    valid_idx, positions = [], []
    for i in range(n_stocks):
        pos = date_pos_list[i].get(date)
        if pos is not None and pos >= W - 1:
            valid_idx.append(i); positions.append(pos)
    N = len(valid_idx)
    if N < 10: continue
    valid_idx = np.array(valid_idx, dtype=np.int32)
    positions = np.array(positions, dtype=np.int32)

    tw = np.zeros((N, W, TDIM_V67), dtype=np.float32)
    lv_arr = np.zeros((N, 6), dtype=np.float32)
    for j, (si, pos) in enumerate(zip(valid_idx, positions)):
        tw[j] = temporal_list[si][pos - W + 1 : pos + 1]
        lv_arr[j] = lastval_list[si][pos]
    tw = norm_t(winsorize(tw))

    cross = np.stack([rank_pct(lv_arr[:, 0]), rank_pct(lv_arr[:, 1]),
                      rank_pct(lv_arr[:, 2]), lv_arr[:, 0] - np.float32(csi_map.get(date, 0.0))], axis=1)
    cross = norm_c(cross); cross_t = np.tile(cross[:, np.newaxis, :], (1, W, 1))
    val = np.stack([rank_pct(lv_arr[:, 3]), rank_pct(lv_arr[:, 4]), rank_pct(lv_arr[:, 5])], axis=1)
    val = norm_c(val); val_t = np.tile(val[:, np.newaxis, :], (1, W, 1))
    feats = np.concatenate([tw, cross_t, val_t], axis=2)

    BS = 4096
    scores_all = []
    for s in range(0, N, BS):
        e = min(s + BS, N)
        bx = torch.from_numpy(feats[s:e]).float().to(DEVICE)
        with torch.no_grad(): sc = model(bx)
        scores_all.append(sc.cpu().numpy())
    scores = np.concatenate(scores_all)

    for j, si in enumerate(valid_idx):
        all_rows.append({"trade_date": date, "ts_code": stock_ids[si], "score": float(scores[j])})

    if (di + 1) % 10 == 0 or di == 0 or di == len(target_dates) - 1:
        print(f"  [{di+1:>3}/{len(target_dates)}] {date}: {N} stk  μ={scores.mean():.3f} σ={scores.std():.3f}")

new_df = pd.DataFrame(all_rows)
if existing is not None and len(existing) > 0:
    existing = existing[~existing["trade_date"].isin(new_df["trade_date"].unique())]
    final = pd.concat([existing, new_df], ignore_index=True)
else:
    final = new_df
final.to_parquet(OUT, index=False)
print(f"[*] {len(final):,} rows saved → {OUT}")
