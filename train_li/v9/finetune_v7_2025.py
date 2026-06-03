"""v9/finetune_v7_2025.py — Fine-tune v7 Spatial on 2025 data, test if IC improves on 2026.

Goal: see if adding 2025 training data helps the model adapt to 2026 regime.
"""
import os, sys, time, gc, glob
import numpy as np, pandas as pd, torch, torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from scipy.stats import spearmanr

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(SCRIPT_DIR)
V2_DIR = os.path.join(ROOT, "v2")
sys.path.insert(0, V2_DIR)
from train import listmle_loss

# ── Model ──
sys.path.insert(0, ROOT)
from v7.models.gru_spatial import GRURankerSpatial

# ── Config ──
WINDOW_SIZE = 60; INPUT_DIM = 26; HIDDEN = 128; LAYERS = 1; DROPOUT = 0.2
D_PROJ = 32; K_ATTN = 5; N_SAMPLE = 1024; GRAD_CLIP = 1.0
CKPT_PATH = os.path.join(ROOT, "v7", "checkpoints",
    "gru_spatial_v7_d32_K5_H128_L1_D0.2_lr0.0003_N1024.pt")
WINDOWS_DIR = os.path.join(ROOT, "processed", "v7_windows")
SCORES_PATH = os.path.join(ROOT, "v7", "results", "daily_scores_spatial_t1.parquet")
PARQUET = os.path.join(ROOT, "processed", "all_data.parquet")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

# ── Dataset for 2025 (val) data, but with caching for training ──
_cache_f = {}; _cache_l = {}
class Year2025Dataset(Dataset):
    def __init__(self, sample_size=None):
        pattern = os.path.join(WINDOWS_DIR, "val", "**", "*.pt")
        self.paths = sorted(glob.glob(pattern, recursive=True))
        if not self.paths:
            raise RuntimeError(f"No .pt files under val/")
        self.sample_size = sample_size
        # Preload
        new = 0
        for path in self.paths:
            if path not in _cache_f:
                data = torch.load(path, map_location="cpu", weights_only=True)
                _cache_f[path] = data["features"].half()
                _cache_l[path] = data["labels"]
                new += 1
        print(f"[2025-ds] Preloaded {len(self.paths)} dates (+{new} new)")

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        path = self.paths[idx]
        features = _cache_f[path][:, -WINDOW_SIZE:, :].float()
        labels = _cache_l[path]
        n = len(labels)
        if self.sample_size is not None and n > self.sample_size:
            perm = torch.randperm(n)[:self.sample_size]
            features = features[perm]
            labels = labels[perm]
        return features, labels

# ── Load model ──
model = GRURankerSpatial(input_dim=INPUT_DIM, hidden_size=HIDDEN,
                          num_layers=LAYERS, dropout=DROPOUT,
                          d_proj=D_PROJ, K=K_ATTN).to(device)
ckpt = torch.load(CKPT_PATH, map_location=device, weights_only=True)
model.load_state_dict(ckpt)
params = sum(p.numel() for p in model.parameters())
print(f"[model] Loaded v7 Spatial ({params:,} params)")

# ── Evaluate IC on 2026 test period ──
def eval_ic(date_range_start, date_range_end):
    """Compute rank IC of model on given date range."""
    df = pd.read_parquet(PARQUET, columns=["trade_date", "ts_code", "pct_chg"])
    df["trade_date"] = df["trade_date"].astype(str)
    df = df[(df["trade_date"] >= date_range_start) & (df["trade_date"] <= date_range_end)]
    ret_d = {}
    for d, sdf in df.groupby("trade_date"):
        ret_d[d] = dict(zip(sdf["ts_code"], pd.to_numeric(sdf["pct_chg"], errors="coerce").astype(float)))

    all_dates = sorted(ret_d.keys())
    # Next-day return mapping
    date_to_next = {all_dates[i]: all_dates[i+1] for i in range(len(all_dates)-1)}

    # We need to load window data for each test date
    # For inference, we need features for all dates
    # Simplified: use precomputed scores from parquet
    # But we want to evaluate the FINE-TUNED model, so we need to run inference
    # This requires loading the window files for test dates

    # Load all test date windows from v7_windows/test/ (doesn't exist) or build on the fly
    # Alternative: build features from raw data
    # For speed, let me just load all windows from the existing processed files
    # We need test dates 2026 January onwards

    # Actually, let me just load ALL window files and filter
    all_pt = glob.glob(os.path.join(WINDOWS_DIR, "**", "*.pt"), recursive=True)
    date_files = {}
    for p in all_pt:
        d = os.path.splitext(os.path.basename(p))[0]
        if d >= date_range_start and d <= date_range_end:
            date_files[d] = p

    ics = []
    model.eval()
    with torch.no_grad():
        for d, path in sorted(date_files.items()):
            if d not in date_to_next: continue
            nd = date_to_next[d]
            if nd not in ret_d: continue

            data = torch.load(path, map_location="cpu", weights_only=True)
            feats = data["features"][:, -WINDOW_SIZE:, :].float().to(device)
            codes = list(data.get("codes", []))
            rr = ret_d[nd]

            if len(feats) == 0: continue
            scores = model(feats).cpu().numpy()

            # Match codes to returns
            if codes:
                s_dict = dict(zip(codes, scores))
                common = [c for c in codes if c in rr]
                if len(common) < 30: continue
                s_arr = np.array([s_dict[c] for c in common])
                r_arr = np.array([rr[c] for c in common])
            else:
                s_arr = scores
                r_arr = np.array([rr.get(str(i), np.nan) for i in range(len(scores))])
                valid = ~np.isnan(r_arr)
                if valid.sum() < 30: continue
                s_arr = s_arr[valid]; r_arr = r_arr[valid]

            ic, _ = spearmanr(s_arr, r_arr)
            if not np.isnan(ic):
                ics.append(ic)

    return ics

print(f"\n[baseline] Computing original v7 IC on 2026 Jan + Apr-May ...")
t0 = time.time()
jan_ic_orig = eval_ic("20260105", "20260130")
apr_ic_orig = eval_ic("20260401", "20260529")
print(f"  Jan IC: mean={np.mean(jan_ic_orig):.4f} std={np.std(jan_ic_orig):.4f} n={len(jan_ic_orig)}")
print(f"  Apr-May IC: mean={np.mean(apr_ic_orig):.4f} std={np.std(apr_ic_orig):.4f} n={len(apr_ic_orig)}")
print(f"  [{time.time()-t0:.0f}s]")

# ── Fine-tune on 2025 ──
print(f"\n[finetune] Training on 2025 data (243 days) ...")
ds = Year2025Dataset(sample_size=N_SAMPLE)
loader = DataLoader(ds, 1, shuffle=True, num_workers=0, pin_memory=True, collate_fn=lambda x: x[0])

# Use a much smaller learning rate for fine-tuning
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5, weight_decay=1e-5)
scaler = torch.amp.GradScaler("cuda", enabled=True)
EPOCHS = 15

for epoch in range(1, EPOCHS + 1):
    model.train()
    loss_sum, n_batch = 0.0, 0
    for feat, lab in loader:
        feat, lab = feat.to(device, non_blocking=True), lab.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast("cuda", enabled=True):
            scores = model(feat)
            loss = listmle_loss(scores, lab)
        if torch.isnan(loss) or torch.isinf(loss): continue
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
        scaler.step(optimizer); scaler.update()
        loss_sum += loss.item(); n_batch += 1

    avg_loss = loss_sum / max(n_batch, 1)
    print(f"  epoch {epoch:2d}/{EPOCHS}  loss={avg_loss:.4f}  batches={n_batch}")
    if n_batch % 50 == 0: torch.cuda.empty_cache()

# ── Evaluate fine-tuned model ──
print(f"\n[finetuned] Computing fine-tuned IC on 2026 ...")
jan_ic_ft = eval_ic("20260105", "20260130")
apr_ic_ft = eval_ic("20260401", "20260529")
print(f"  Jan IC: mean={np.mean(jan_ic_ft):.4f} std={np.std(jan_ic_ft):.4f} n={len(jan_ic_ft)}")
print(f"  Apr-May IC: mean={np.mean(apr_ic_ft):.4f} std={np.std(apr_ic_ft):.4f} n={len(apr_ic_ft)}")

# ── Comparison ──
print(f"\n{'='*60}")
print(f" COMPARISON")
print(f"{'='*60}")
print(f"  {'Period':<12} {'Original v7':>12} {'Fine-tuned':>12} {'Delta':>10}")
ml = np.mean(jan_ic_orig); mf = np.mean(jan_ic_ft)
print(f"  {'Jan 2026':<12} {ml:>+11.4f} {mf:>+11.4f} {mf-ml:>+9.4f}")
ml = np.mean(apr_ic_orig); mf = np.mean(apr_ic_ft)
print(f"  {'Apr-May 2026':<12} {ml:>+11.4f} {mf:>+11.4f} {mf-ml:>+9.4f}")

# Save fine-tuned checkpoint
ft_path = os.path.join(SCRIPT_DIR, "checkpoints", "v7_finetuned_2025.pt")
os.makedirs(os.path.dirname(ft_path), exist_ok=True)
torch.save(model.state_dict(), ft_path)
print(f"\nSaved: {ft_path}")
