"""v9/eval_finetune.py — Fine-tune v7 on 2025, then evaluate IC on 2026 Jan + Apr-May.

Uses the same feature pipeline as trade/infer.py for 2026 dates.
"""
import os, sys, time, glob, gc
import numpy as np, pandas as pd, torch, torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from scipy.stats import spearmanr

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(SCRIPT_DIR)
V2_DIR = os.path.join(ROOT, "v2")
sys.path.insert(0, V2_DIR)
from train import listmle_loss

sys.path.insert(0, ROOT)
from v7.models.gru_spatial import GRURankerSpatial

# ── Config ──
WINDOW_SIZE = 60; INPUT_DIM = 26
HIDDEN = 128; LAYERS = 1; DROPOUT = 0.2; D_PROJ = 32; K_ATTN = 5
N_SAMPLE = 1024; GRAD_CLIP = 1.0; LR_FT = 1e-4; EPOCHS = 15  # More aggressive
CKPT = os.path.join(ROOT, "v7", "checkpoints",
    "gru_spatial_v7_d32_K5_H128_L1_D0.2_lr0.0003_N1024.pt")
WINDOWS_DIR = os.path.join(ROOT, "processed", "v7_windows")
DATA_DIR = os.path.join(ROOT, "data", "daily")
MARKET_DIR = os.path.join(ROOT, "data", "market")
ALL_DATA = os.path.join(ROOT, "processed", "all_data.parquet")
INDEX_P = os.path.join(MARKET_DIR, "000300.SH.csv")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")
t0 = time.time()

# ═══════════════════════════════════════════════════════
# 1. Load + Fine-tune v7 on 2025
# ═══════════════════════════════════════════════════════
_cache_f, _cache_l = {}, {}

class Year2025DS(Dataset):
    def __init__(self, sample_size=N_SAMPLE):
        pattern = os.path.join(WINDOWS_DIR, "val", "**", "*.pt")
        self.paths = sorted(glob.glob(pattern, recursive=True))
        print(f"[ds] Found {len(self.paths)} dates in val/")
        self.sample_size = sample_size
        new = 0
        for p in self.paths:
            if p not in _cache_f:
                d = torch.load(p, map_location="cpu", weights_only=True)
                _cache_f[p] = d["features"].half()
                _cache_l[p] = d["labels"]
                new += 1
        print(f"[ds] Preloaded {len(self.paths)} dates (+{new} new)")

    def __len__(self): return len(self.paths)

    def __getitem__(self, idx):
        p = self.paths[idx]
        f = _cache_f[p][:, -WINDOW_SIZE:, :].float()
        l = _cache_l[p]
        n = len(l)
        if self.sample_size and n > self.sample_size:
            perm = torch.randperm(n)[:self.sample_size]
            f, l = f[perm], l[perm]
        return f, l

print("\n[1] Loading v7 checkpoint...")
model = GRURankerSpatial(input_dim=INPUT_DIM, hidden_size=HIDDEN,
                          num_layers=LAYERS, dropout=DROPOUT,
                          bidirectional=False, d_proj=D_PROJ, K=K_ATTN).to(device)
model.load_state_dict(torch.load(CKPT, map_location=device, weights_only=True))

print("[2] Fine-tuning on 2025 data...")
ds = Year2025DS()
loader = DataLoader(ds, 1, shuffle=True, num_workers=0, pin_memory=True, collate_fn=lambda x: x[0])
opt = torch.optim.AdamW(model.parameters(), lr=LR_FT, weight_decay=1e-5)
scaler = torch.amp.GradScaler("cuda", enabled=True)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="min", factor=0.5, patience=3)

best_loss = float("inf"); best_state = None
for ep in range(1, EPOCHS + 1):
    model.train(); ls, nb = 0.0, 0
    for feat, lab in loader:
        feat, lab = feat.to(device, non_blocking=True), lab.to(device, non_blocking=True)
        opt.zero_grad(set_to_none=True)
        with torch.amp.autocast("cuda", enabled=True):
            scores = model(feat)
            loss = listmle_loss(scores, lab)
        if torch.isnan(loss) or torch.isinf(loss): continue
        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
        scaler.step(opt); scaler.update()
        ls += loss.item(); nb += 1
    avg = ls / max(nb, 1)
    scheduler.step(avg)
    marker = "*" if avg < best_loss else " "
    if avg < best_loss:
        best_loss = avg; best_state = {k: v.clone() for k, v in model.state_dict().items()}
    print(f"  ep {ep:2d}  loss={avg:.4f} {marker}  lr={opt.param_groups[0]['lr']:.1e}")
    if nb % 50 == 0: torch.cuda.empty_cache()

model.load_state_dict(best_state)
ft_path = os.path.join(SCRIPT_DIR, "checkpoints", "v7_finetuned_2025.pt")
os.makedirs(os.path.dirname(ft_path), exist_ok=True)
torch.save(model.state_dict(), ft_path)
print(f"  Saved: {ft_path}")

# ═══════════════════════════════════════════════════════
# 2. Evaluate IC on 2026 using trade/infer.py feature pipeline
# ═══════════════════════════════════════════════════════
print(f"\n[3] Evaluating IC on 2026 test dates... [{time.time()-t0:.0f}s]")

# We can use the EXISTING v7 scores parquet for 2026 days
# The scores were produced by running v7 model on features computed from daily CSV data
# To get fine-tuned model scores, we'd need to recompute features
# 
# Shortcut: we know the features used for each date in the v7 scores parquet
# But we don't have the raw input tensors stored.
#
# Alternative: load the all_data.parquet for 2026 dates
# The parquet has the raw features needed for inference
# We can compute features from parquet (same logic as build_windows but for single dates)

# Load 2026 return data
df_all = pd.read_parquet(ALL_DATA, columns=["trade_date", "ts_code", "pct_chg",
    "open", "high", "low", "close", "vwap", "vol", "amount", "turnover_rate",
    "volume_ratio", "total_mv", "pe", "pb", "circ_mv"])
df_all["trade_date"] = df_all["trade_date"].astype(str)
df_all = df_all[df_all["trade_date"] >= "20251101"]  # need T lookback
print(f"  Loaded {len(df_all)} rows from all_data.parquet")
# Check if vwap column is all NaN
vwap_ok = df_all["vwap"].notna().any()
print(f"  vwap available: {vwap_ok}")

# Per-stock DataFrames for 60-day lookback
stocks = {}
for code, sdf in df_all.groupby("ts_code"):
    sdf = sdf.sort_values("trade_date")
    if len(sdf) >= WINDOW_SIZE + 2:
        stocks[code] = sdf
print(f"  {len(stocks)} stocks with >= 62 days")

all_dates_raw = sorted(df_all["trade_date"].unique())
test_dates = [d for d in all_dates_raw if "2026" in d and d <= "20260529"]
# Subsample to speed up: 5 dates per month
import random; random.seed(42)
sampled = {}
for d in test_dates:
    sampled.setdefault(d[:6], []).append(d)
test_sample = []
for m in sorted(sampled.keys()):
    test_sample.extend(random.sample(sampled[m], min(5, len(sampled[m]))))
test_dates = sorted(test_sample)
print(f"  Sampled {len(test_dates)} dates for evaluation ({len(test_sample)//5} per month)")
print(f"  {len(test_dates)} test dates")

# Feature computation (simplified from trade/infer.py)
CSI = pd.read_csv(INDEX_P, dtype={"trade_date": str})
csi_m = dict(zip(CSI["trade_date"], pd.to_numeric(CSI["pct_chg"], errors="coerce") / 100.0))
LABEL_HORIZON = 1

BASIC = ["open", "high", "low", "close", "vol", "amount", "pct_chg",
         "turnover_rate", "volume_ratio", "total_mv"]
LOG_COLS = {"vol", "amount", "total_mv"}

def add_tech(sdf):
    c = sdf["close"].astype(float)
    e12 = c.ewm(span=12, adjust=False).mean()
    e26 = c.ewm(span=26, adjust=False).mean()
    sdf["macd"] = e12 - e26
    sdf["macd_signal"] = sdf["macd"].ewm(span=9, adjust=False).mean()
    d = c.diff()
    g = d.clip(lower=0); l = (-d).clip(lower=0)
    rs = g.rolling(14).mean() / (l.rolling(14).mean() + 1e-8)
    sdf["rsi"] = 100 - 100 / (1 + rs)
    ma20 = c.rolling(20).mean(); std20 = c.rolling(20).std()
    sdf["bb_width"] = (2 * std20) / (ma20 + 1e-8)
    sdf["bb_pct"] = (c - (ma20 - 2 * std20)) / (4 * std20 + 1e-8)
    sdf["mom_5"] = c / c.shift(5) - 1
    sdf["mom_20"] = c / c.shift(20) - 1
    sdf["vol_20"] = c.pct_change().rolling(20).std()
    return sdf

TECH = ["macd", "macd_signal", "rsi", "bb_width", "bb_pct", "mom_5", "mom_20", "vol_20"]
PE_CLIP = (0.1, 500.0); PB_CLIP = (0.1, 50.0)

def rank_pct(arr):
    valid = ~np.isnan(arr)
    out = np.zeros(len(arr), dtype=np.float32)
    if valid.sum() >= 2:
        o = np.argsort(np.argsort(arr[valid]))
        out[valid] = o.astype(np.float32) / max(valid.sum() - 1, 1)
    return out

def norm_temporal(arr):
    m, s = arr.mean(axis=0, keepdims=True), arr.std(axis=0, keepdims=True) + 1e-8
    return (arr - m) / s

def norm_cross(arr):
    m, s = arr.mean(axis=0, keepdims=True), arr.std(axis=0, keepdims=True) + 1e-8
    return (arr - m) / s

def winsorize(arr, pl=1, ph=99):
    lo = np.percentile(arr, pl, axis=0, keepdims=True)
    hi = np.percentile(arr, ph, axis=0, keepdims=True)
    return np.clip(arr, lo, hi)

# Evaluate IC per date
monthly_ics = {}
model.eval()

for di, td in enumerate(test_dates):
    # Need next day for label
    td_idx = all_dates_raw.index(td)
    if td_idx + LABEL_HORIZON >= len(all_dates_raw): continue
    nd = all_dates_raw[td_idx + 1]

    # Collect features for all stocks on this date
    feats_list = []; codes_list = []
    cross_pct = []; cross_amt = []; cross_tvr = []; cross_pe = []; cross_pb = []; cross_cm = []

    for code, sdf in stocks.items():
        # Get 60-day window ending at td
        sdf_td = sdf[sdf["trade_date"] <= td]
        if len(sdf_td) < WINDOW_SIZE: continue
        w = sdf_td.iloc[-WINDOW_SIZE:].copy()

        # Check if next day exists
        nd_rows = sdf[sdf["trade_date"] == nd]
        if len(nd_rows) == 0: continue

        w = add_tech(w)
        # Raw temporal
        raw = w[BASIC].astype(float).values.copy()  # [T, 10]
        for lc in LOG_COLS:
            if lc in BASIC:
                idx = BASIC.index(lc)
                raw[:, idx] = np.log1p(np.maximum(raw[:, idx], 0))
        tech = w[TECH].astype(float).values.copy()  # [T, 8]
        vwap_g = (w["close"].values / np.maximum(w.get("vwap", w["close"]).values, 1e-8) - 1)  # [T]
        temporal = np.concatenate([raw, tech, vwap_g[:, None]], axis=1)  # [T, 19]
        feats_list.append(temporal)
        codes_list.append(code)

        # Cross-sectional snapshots (last row)
        lr = w.iloc[-1]
        cross_pct.append(float(lr.get("pct_chg", 0)))
        cross_amt.append(float(lr.get("amount", 0)))
        cross_tvr.append(float(lr.get("turnover_rate", 0)))
        cross_pe.append(float(lr.get("pe", 0)))
        cross_pb.append(float(lr.get("pb", 0)))
        cross_cm.append(float(lr.get("circ_mv", 0)))

    N = len(codes_list)
    if N < 30: continue

    # Build cross-sectional features
    pct_a = np.array(cross_pct, dtype=np.float32)
    amt_a = np.array(cross_amt, dtype=np.float32)
    tvr_a = np.array(cross_tvr, dtype=np.float32)
    pe_a = np.array(cross_pe, dtype=np.float32)
    pb_a = np.array(cross_pb, dtype=np.float32)
    cm_a = np.array(cross_cm, dtype=np.float32)

    c_feat = np.zeros((N, 7), dtype=np.float32)
    c_feat[:, 0] = rank_pct(pct_a)
    c_feat[:, 1] = rank_pct(amt_a)
    c_feat[:, 2] = rank_pct(tvr_a)
    c_feat[:, 3] = pct_a - csi_m.get(td, 0)
    c_feat[:, 4] = rank_pct(np.clip(pe_a, PE_CLIP[0], PE_CLIP[1]))
    c_feat[:, 5] = rank_pct(np.clip(pb_a, PB_CLIP[0], PB_CLIP[1]))
    c_feat[:, 6] = rank_pct(cm_a)
    c_norm = norm_cross(c_feat)

    # Winsorize + normalize temporal
    ts = np.stack(feats_list, axis=0)  # [N, T, 19]
    Nt, Tt, Ft = ts.shape
    t_flat = ts.reshape(-1, Ft)
    t_flat = winsorize(t_flat)
    ts = t_flat.reshape(Nt, Tt, Ft)

    final = np.zeros((N, 60, 26), dtype=np.float32)
    for i in range(N):
        tn = norm_temporal(ts[i])
        ct = np.tile(c_norm[i], (60, 1))
        final[i] = np.concatenate([tn, ct], axis=-1)

    # Inference
    batch = torch.from_numpy(final).float().to(device)
    with torch.no_grad():
        scores = model(batch).cpu().numpy()

    # Get next-day returns
    nd_rets = []
    nd_scores = []
    for i, code in enumerate(codes_list):
        nr = stocks[code][stocks[code]["trade_date"] == nd]
        if len(nr) > 0:
            nd_rets.append(float(nr.iloc[0]["pct_chg"]))
            nd_scores.append(scores[i])

    if len(nd_rets) < 30: continue
    ic, _ = spearmanr(nd_scores, nd_rets)
    if np.isnan(ic): continue

    m = td[:6]
    monthly_ics.setdefault(m, []).append(ic)

    if (di+1) % 20 == 0:
        print(f"  [{time.time()-t0:.0f}s] {di+1}/{len(test_dates)}: {td}  N={N}  IC={ic:+.4f}")

print(f"\n{'='*60}")
print(f" Fine-tuned v7 (on 2025) — 2026 Monthly IC")
print(f"{'='*60}")

# Original v7 IC from earlier analysis (for comparison)
orig_ic = {"202601": 0.1238, "202602": 0.0656, "202603": 0.1169, "202604": -0.0062, "202605": 0.0131}

print(f"  {'Month':<8} {'#Days':>6} {'FT_IC':>8} {'Orig_IC':>8} {'Delta':>8}")
print(f"  {'─'*48}")
for m in sorted(monthly_ics.keys()):
    ics = monthly_ics[m]
    oic = orig_ic.get(m, 0)
    mic = np.mean(ics)
    print(f"  {m:<8} {len(ics):>6} {mic:>+7.4f} {oic:>+7.4f} {mic-oic:>+7.4f}")

print(f"\n  Total time: {time.time()-t0:.0f}s")
