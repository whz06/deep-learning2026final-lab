"""
v2/ensemble.py — Heterogeneous ensemble (weighted fusion) + homogeneous seed ensemble.

Stage 1: Save per-date score vectors from 3 frozen models on val set.
Stage 2: Grid-search fusion weights (alpha,beta,gamma) on val set.
Stage 3: Train best model with N random seeds, save averaged scores.

Usage:
  python ensemble.py --stage save_scores
  python ensemble.py --stage sweep
  python ensemble.py --stage seeds --model gru
"""
import os, sys, re, json, itertools, gc
import numpy as np
import torch
from torch.utils.data import DataLoader

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CKPT_DIR  = os.path.join(SCRIPT_DIR, "checkpoints")
SCORE_DIR = os.path.join(SCRIPT_DIR, "scores")
RESULTS_CSV = os.path.join(SCRIPT_DIR, "results.csv")
os.makedirs(SCORE_DIR, exist_ok=True)

from config import WINDOW_SIZE, ENSEMBLE_SEEDS
from dataset import DailyStockDataset
from models import MLPRanker, GRURanker, TransformerRanker


# ── Best checkpoint names from sweep results ──
BEST_CKPTS = {
    "gru": "gru_gru_hidden_size=128_num_layers=1_dropout=0.2_lr=0.0003.pt",
    "tf":  "tf_tf_d_model=96_n_heads=4_n_temporal_layers=2_n_spatial_layers=1_dropout=0.1_lr=0.0003.pt",
    "mlp": "mlp_mlp_hidden_dim=1024_n_layers=4_dropout=0.3_lr=0.0005.pt",
}


def parse_cfg(mtype, ckpt_name):
    """Extract model params from checkpoint filename."""
    name = ckpt_name.replace(".pt", "")
    if mtype == "gru":
        return {
            "input_dim": 22,
            "hidden_size": int(re.search(r"hidden_size=(\d+)", name).group(1)),
            "num_layers": int(re.search(r"num_layers=(\d+)", name).group(1)),
            "dropout": float(re.search(r"dropout=([\d.]+)_lr", name).group(1)),
        }
    elif mtype == "mlp":
        return {
            "input_dim": 22 * WINDOW_SIZE,
            "hidden_dim": int(re.search(r"hidden_dim=(\d+)", name).group(1)),
            "n_layers": int(re.search(r"n_layers=(\d+)", name).group(1)),
            "dropout": float(re.search(r"dropout=([\d.]+)_lr", name).group(1)),
        }
    elif mtype == "tf":
        return {
            "input_dim": 22,
            "d_model": int(re.search(r"d_model=(\d+)", name).group(1)),
            "n_heads": int(re.search(r"n_heads=(\d+)", name).group(1)),
            "n_temporal_layers": int(re.search(r"n_temporal_layers=(\d+)", name).group(1)),
            "n_spatial_layers": int(re.search(r"n_spatial_layers=(\d+)", name).group(1)),
            "dropout": float(re.search(r"dropout=([\d.]+)_lr", name).group(1)),
        }
    return {}


def load_model(mtype, ckpt_name, device):
    """Load model from checkpoint."""
    params = parse_cfg(mtype, ckpt_name)
    Model = {"mlp": MLPRanker, "gru": GRURanker, "tf": TransformerRanker}[mtype]
    model = Model(**params)
    ckpt_path = os.path.join(CKPT_DIR, ckpt_name)
    model.load_state_dict(torch.load(ckpt_path, map_location=device, weights_only=True))
    model.to(device)
    model.eval()
    return model


# ── Stage 1: Save score vectors ──

def save_scores(device):
    """Run all 3 models on val set, save score dicts as .npz."""
    val_ds = DailyStockDataset("val", WINDOW_SIZE, sample_size=None, shuffle=False)
    val_loader = DataLoader(val_ds, 1, shuffle=False, num_workers=0,
                            pin_memory=True, collate_fn=lambda x: x[0])

    for mtype, ckpt_name in BEST_CKPTS.items():
        print(f"[save_scores] {mtype} from {ckpt_name}")
        model = load_model(mtype, ckpt_name, device)

        date_scores = {}
        with torch.no_grad():
            for d_idx, (feat, lab) in enumerate(val_loader):
                feat = feat.to(device)
                scores = model(feat).cpu().numpy()
                date_str = os.path.basename(val_ds.file_paths[d_idx]).replace(".pt", "")
                date_scores[date_str] = scores.astype(np.float32)
                torch.cuda.empty_cache()

        out_path = os.path.join(SCORE_DIR, f"{mtype}_scores.npz")
        np.savez_compressed(out_path, **date_scores)
        print(f"  Saved {len(date_scores)} dates → {out_path}")
        del model
        torch.cuda.empty_cache()


# ── Stage 2: Grid-search fusion weights ──

def sweep_weights(device):
    """Sweep alpha+beta+gamma=1 on saved val scores."""
    print("[sweep] Loading score files ...")
    score_dicts = {}
    for mtype in BEST_CKPTS:
        path = os.path.join(SCORE_DIR, f"{mtype}_scores.npz")
        data = np.load(path)
        score_dicts[mtype] = {k: data[k] for k in data.files}
        print(f"  {mtype}: {len(data.files)} dates")

    # Load val labels
    val_ds = DailyStockDataset("val", WINDOW_SIZE, sample_size=None, shuffle=False)
    val_loader = DataLoader(val_ds, 1, shuffle=False, num_workers=0,
                            pin_memory=True, collate_fn=lambda x: x[0])
    date_labels = {}
    for d_idx, (feat, lab) in enumerate(val_loader):
        date_str = os.path.basename(val_ds.file_paths[d_idx]).replace(".pt", "")
        date_labels[date_str] = lab.numpy()

    models = ["gru", "tf", "mlp"]
    steps = int(1.0 / 0.1)
    best = {"alpha": 0, "beta": 0, "gamma": 0, "ic": -999}

    for a_i in range(steps + 1):
        for b_i in range(steps + 1 - a_i):
            alpha = a_i * 0.1
            beta  = b_i * 0.1
            gamma = 1.0 - alpha - beta

            daily_ics = []
            for date, labels in date_labels.items():
                s = [score_dicts[m].get(date) for m in models]
                if any(x is None for x in s):
                    continue
                fused = alpha * s[0] + beta * s[1] + gamma * s[2]
                r_f = np.argsort(np.argsort(fused)).astype(float)
                r_l = np.argsort(np.argsort(labels)).astype(float)
                r_f -= r_f.mean(); r_l -= r_l.mean()
                denom = np.linalg.norm(r_f) * np.linalg.norm(r_l)
                ic = 0.0 if denom < 1e-12 else (r_f * r_l).sum() / denom
                daily_ics.append(ic)

            avg_ic = float(np.mean(daily_ics)) if daily_ics else 0.0
            if avg_ic > best["ic"]:
                best = {"alpha": alpha, "beta": beta, "gamma": gamma, "ic": avg_ic}

    print(f"\n[sweep] Best fusion weights:")
    print(f"  alpha(gru)={best['alpha']:.1f}  beta(tf)={best['beta']:.1f}  "
          f"gamma(mlp)={best['gamma']:.1f}")
    print(f"  Val IC={best['ic']:.4f}  (vs single GRU IC=0.1029)")

    with open(os.path.join(SCORE_DIR, "best_weights.json"), "w") as f:
        json.dump(best, f, indent=2)
    return best


# ── Stage 3: Seed ensemble ──

def train_seed_models(model_type, device):
    """Train BEST_CKPTS[model_type] config with multiple seeds."""
    ckpt_name = BEST_CKPTS[model_type]
    params = parse_cfg(model_type, ckpt_name)
    Model = {"mlp": MLPRanker, "gru": GRURanker, "tf": TransformerRanker}[model_type]
    lr = float(re.search(r"lr=([\d.eE+-]+?)(?:_|\.pt)", ckpt_name).group(1))

    print(f"[seeds] Training {model_type} × {len(ENSEMBLE_SEEDS)} seeds")

    from train import train_one
    from config import GRUConfig, MLPConfig, TFConfig
    from config import EPOCHS, PATIENCE, GRAD_CLIP, SCHEDULER_FACTOR, SCHEDULER_PATIENCE

    # Build config object
    if model_type == "gru":
        cfg = GRUConfig(**{**params, "lr": lr})
    elif model_type == "mlp":
        cfg = MLPConfig(**{**params, "lr": lr})
    elif model_type == "tf":
        cfg = TFConfig(**{**params, "lr": lr})

    for seed in ENSEMBLE_SEEDS:
        ckpt_path = os.path.join(CKPT_DIR, f"{model_type}_seed{seed}.pt")
        if os.path.exists(ckpt_path):
            print(f"  Seed {seed}: checkpoint exists, skip")
            continue

        print(f"  Seed {seed}: training ...")
        torch.manual_seed(seed)
        np.random.seed(seed)

        model = Model(**params).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="max", factor=SCHEDULER_FACTOR, patience=SCHEDULER_PATIENCE)

        train_ds = DailyStockDataset("train", WINDOW_SIZE, sample_size=2048, shuffle=True)
        val_ds   = DailyStockDataset("val",   WINDOW_SIZE, sample_size=None, shuffle=False)
        train_loader = DataLoader(train_ds, 1, shuffle=True, num_workers=0, pin_memory=True, collate_fn=lambda x: x[0])
        val_loader   = DataLoader(val_ds,   1, shuffle=False, num_workers=0, pin_memory=True, collate_fn=lambda x: x[0])

        best_ic, best_ep, no_imp = -float("inf"), 0, 0
        for epoch in range(1, EPOCHS + 1):
            model.train()
            for _, (feat, lab) in enumerate(train_loader):
                feat, lab = feat.to(device), lab.to(device)
                optimizer.zero_grad(set_to_none=True)
                scores = model(feat)

                # ListMLE loss
                _, idx = torch.sort(lab, descending=True)
                s = scores[idx]
                log_cumsum = torch.logcumsumexp(s.flip(0), dim=0).flip(0)
                loss = -(s - log_cumsum)[:-1].sum()

                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
                optimizer.step()

            model.eval()
            torch.cuda.empty_cache()
            val_ics = []
            with torch.no_grad():
                for feat, lab in val_loader:
                    feat, lab = feat.to(device), lab.to(device)
                    rs = torch.argsort(torch.argsort(model(feat))).float()
                    rl = torch.argsort(torch.argsort(lab)).float()
                    rs -= rs.mean(); rl -= rl.mean()
                    denom = rs.norm() * rl.norm()
                    ic = torch.tensor(0.0) if denom < 1e-12 else (rs * rl).sum() / denom
                    val_ics.append(ic.item())
            torch.cuda.empty_cache()

            avg_ic = float(np.mean(val_ics))
            scheduler.step(avg_ic)
            if avg_ic > best_ic:
                best_ic, best_ep, no_imp = avg_ic, epoch, 0
                torch.save(model.state_dict(), ckpt_path)
            else:
                no_imp += 1
            if no_imp >= PATIENCE:
                print(f"    Early stop @ {epoch}, best IC={best_ic:.4f}")
                break

        del model, optimizer, scheduler
        gc = __import__("gc"); gc.collect()
        torch.cuda.empty_cache()

    print(f"[seeds] Done. {len(ENSEMBLE_SEEDS)} seed checkpoints saved.")


# ── Main ──

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", required=True, choices=["save_scores","sweep","seeds"])
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--model", default="gru")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args.stage == "save_scores":
        save_scores(device)
    elif args.stage == "sweep":
        sweep_weights(device)
    elif args.stage == "seeds":
        train_seed_models(args.model, device)
