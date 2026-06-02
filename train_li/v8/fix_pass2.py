"""v8/fix_pass2.py — Run only Pass 2 (winsorization + normalize) on existing .pt files."""
import os, sys, numpy as np, torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
WINDOWS_DIR = os.path.join(ROOT, "processed", "v8_windows")
N_TEMPORAL = 24
PE_CLIP = (0.1, 500.0); PB_CLIP = (0.1, 50.0)

def winsorize_2d(arr, p_low=1, p_high=99):
    lo = np.percentile(arr, p_low, axis=0, keepdims=True)
    hi = np.percentile(arr, p_high, axis=0, keepdims=True)
    return np.clip(arr, lo, hi)

def rank_percentile(arr):
    order = np.argsort(np.argsort(arr))
    return order.astype(np.float32) / max(len(arr) - 1, 1)

def normalize_temporal(arr):
    mean = arr.mean(axis=0, keepdims=True)
    std = arr.std(axis=0, keepdims=True) + 1e-8
    return (arr - mean) / std

def normalize_cross(arr):
    mean = arr.mean(axis=0, keepdims=True)
    std = arr.std(axis=0, keepdims=True) + 1e-8
    return (arr - mean) / std

all_pt = []
for root, dirs, files in os.walk(WINDOWS_DIR):
    for f in files:
        if f.endswith(".pt"):
            all_pt.append(os.path.join(root, f))
all_pt = sorted(all_pt)
print(f"[fix] Checking {len(all_pt)} pt files ...")

skip_count, fix_count = 0, 0
for i, pt_path in enumerate(all_pt):
    data = torch.load(pt_path, map_location="cpu", weights_only=True)

    # Already has 'features' key -> Pass 2 done
    if "features" in data:
        skip_count += 1
        continue

    # Needs Pass 2 processing
    fix_count += 1
    temporal_raw = data["temporal"].numpy()    # [N, T, 24]
    labels = data["labels"].numpy()
    cross_raw = data["cross_raw"].numpy()       # [N, 7]
    industry_id = data.get("industry_id", torch.zeros(len(labels), dtype=torch.long))
    T = temporal_raw.shape[1]
    N = len(labels)
    if N < 2: continue

    temporal_flat = temporal_raw.reshape(-1, N_TEMPORAL)
    temporal_flat = winsorize_2d(temporal_flat, p_low=1, p_high=99)
    temporal_raw = temporal_flat.reshape(N, T, -1)

    temporal_norm = np.zeros_like(temporal_raw)
    for j in range(N):
        temporal_norm[j] = normalize_temporal(temporal_raw[j])

    cross_feat = np.zeros((N, 7), dtype=np.float32)
    for col_idx in [0, 1, 2, 4, 5, 6]:
        raw_vals = cross_raw[:, col_idx].copy()
        valid = ~np.isnan(raw_vals)
        if valid.sum() < 2:
            cross_feat[:, col_idx] = 0.0
        else:
            if col_idx == 4: raw_vals = np.clip(raw_vals, *PE_CLIP)
            elif col_idx == 5: raw_vals = np.clip(raw_vals, *PB_CLIP)
            ranks = np.zeros(N, dtype=np.float32)
            valid_clip = ~np.isnan(raw_vals)
            if valid_clip.sum() >= 2:
                ranks[valid_clip] = rank_percentile(raw_vals[valid_clip])
            cross_feat[:, col_idx] = ranks
    cross_feat[:, 3] = cross_raw[:, 0] - cross_raw[:, 3]

    cross_norm = normalize_cross(cross_feat)
    cross_tiled = np.tile(cross_norm[:, np.newaxis, :], (1, T, 1))
    features_full = np.concatenate([temporal_norm, cross_tiled], axis=-1)

    torch.save({
        "features": torch.from_numpy(features_full).float(),
        "labels": torch.from_numpy(labels).float(),
        "ts_codes": data["ts_codes"],
        "industry_id": industry_id,
    }, pt_path)

    if fix_count % 100 == 0:
        print(f"  [fix] {fix_count} fixed, {skip_count} skipped ...")

print(f"[fix] Done. {fix_count} fixed, {skip_count} already had 'features'.")
