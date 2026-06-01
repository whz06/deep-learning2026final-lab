"""
v2/train.py — Model training + sweep for MLP / GRU / Transformer.

Usage:
  python train.py --model mlp --sweep               # MLP grid sweep
  python train.py --model gru --sweep               # GRU fine-tune sweep
  python train.py --model tf  --sweep phase1        # TF coarse sweep
  python train.py --model tf  --sweep phase2        # TF fine sweep
  python train.py --model mlp --best                # Train best MLP
"""
import os, sys, gc, time, json, itertools, argparse
import numpy as np
import pandas as pd
import torch, torch.nn as nn
from torch.utils.data import DataLoader

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF",
                      "max_split_size_mb:128,roundup_power2_divisions:16,garbage_collection_threshold:0.6")

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(SCRIPT_DIR)

from config import (
    WINDOW_SIZE, INPUT_DIM, BATCH_SIZE, BATCH_SIZE_MLP, EPOCHS, PATIENCE,
    GRAD_CLIP, SCHEDULER_FACTOR, SCHEDULER_PATIENCE,
    MLPConfig, GRUConfig, TFConfig,
    MLP_SWEEP, GRU_SWEEP, TF_SWEEP_COARSE, TF_SWEEP_FINE,
)
from dataset import DailyStockDataset
from models import MLPRanker, GRURanker, TransformerRanker

CKPT_DIR   = os.path.join(SCRIPT_DIR, "checkpoints")
RESULTS_CSV = os.path.join(SCRIPT_DIR, "results.csv")
os.makedirs(CKPT_DIR, exist_ok=True)

# ── Losses ──

def listmle_loss(scores, labels):
    _, idx = torch.sort(labels, descending=True)
    s = scores[idx]
    log_cumsum = torch.logcumsumexp(s.flip(0), dim=0).flip(0)
    return -(s - log_cumsum)[:-1].sum()

@torch.no_grad()
def rankic(scores, labels):
    def trank(x): return torch.argsort(torch.argsort(x)).float()
    rs, rl = trank(scores), trank(labels)
    rs, rl = rs - rs.mean(), rl - rl.mean()
    denom = rs.norm() * rl.norm()
    return torch.tensor(0.0) if denom < 1e-12 else (rs * rl).sum() / denom

# ── Model factory ──

def build_model(model_type: str, cfg):
    if model_type == "mlp":
        return MLPRanker(cfg.input_dim, cfg.hidden_dim, cfg.n_layers, cfg.dropout)
    elif model_type == "gru":
        return GRURanker(cfg.input_dim, cfg.hidden_size, cfg.num_layers, cfg.dropout)
    elif model_type == "tf":
        return TransformerRanker(cfg.input_dim, cfg.d_model, cfg.n_heads,
                                  cfg.n_temporal_layers, cfg.n_spatial_layers, cfg.dropout)
    raise ValueError(model_type)

# ── Training ──

def train_one(model_type: str, cfg, device, verbose=True, save_best=True):
    name = f"{model_type}_{_cfg_name(model_type, cfg)}"
    bs = BATCH_SIZE_MLP if model_type == "mlp" else BATCH_SIZE
    print(f"\n{'='*60}\nTraining: {name}\n  cfg={cfg}\n{'='*60}")

    train_ds = DailyStockDataset("train", WINDOW_SIZE, sample_size=bs, shuffle=True)
    val_ds   = DailyStockDataset("val",   WINDOW_SIZE, sample_size=None, shuffle=False)
    train_loader = DataLoader(train_ds, 1, shuffle=True, num_workers=0,
                              pin_memory=True, collate_fn=lambda x: x[0])
    val_loader   = DataLoader(val_ds,   1, shuffle=False, num_workers=0,
                              pin_memory=True, collate_fn=lambda x: x[0])

    model = build_model(model_type, cfg).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=SCHEDULER_FACTOR, patience=SCHEDULER_PATIENCE)

    best_ic, best_ep, no_imp = -float("inf"), 0, 0
    t0 = time.time()

    for epoch in range(1, EPOCHS + 1):
        model.train()
        loss_sum, n_batch = 0.0, 0
        for _, (feat, lab) in enumerate(train_loader):
            feat, lab = feat.to(device), lab.to(device)
            optimizer.zero_grad(set_to_none=True)
            scores = model(feat)
            loss = listmle_loss(scores, lab)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            optimizer.step()
            loss_sum += loss.item(); n_batch += 1
            if n_batch % 100 == 0:
                torch.cuda.empty_cache()

        model.eval()
        torch.cuda.empty_cache()
        val_ics = []
        with torch.no_grad():
            for feat, lab in val_loader:
                feat, lab = feat.to(device), lab.to(device)
                ic = rankic(model(feat), lab)
                val_ics.append(ic.item())
        torch.cuda.empty_cache()

        avg_ic = float(np.mean(val_ics)) if val_ics else 0.0
        scheduler.step(avg_ic)

        if avg_ic > best_ic:
            best_ic, best_ep, no_imp = avg_ic, epoch, 0
            if save_best:
                torch.save(model.state_dict(), os.path.join(CKPT_DIR, f"{name}.pt"))
        else:
            no_imp += 1

        if verbose and (epoch % 5 == 0 or epoch == 1):
            print(f"  ep {epoch:3d} | loss {loss_sum/max(n_batch,1):.4f} | "
                  f"val_ic {avg_ic:.4f} | best {best_ic:.4f} @ {best_ep} | "
                  f"lr {optimizer.param_groups[0]['lr']:.2e}")

        if no_imp >= PATIENCE:
            if verbose: print(f"  Early stop @ {epoch}")
            break

    elapsed = time.time() - t0
    result = {"name": name, "best_val_rankic": round(best_ic, 6),
              "best_epoch": best_ep, "total_epochs": epoch, "time_s": round(elapsed, 1),
              **{k: v for k, v in cfg.__dict__.items()}}
    _append_csv(result)
    if verbose: print(f"  -> IC={best_ic:.4f} @ ep{best_ep} ({elapsed:.0f}s)")

    del model, optimizer, scheduler, train_loader, val_loader
    gc.collect(); torch.cuda.empty_cache()
    return result

# ── Sweep ──

def _cfg_name(mtype, cfg):
    parts = [mtype]
    for k, v in cfg.__dict__.items():
        if k in ("input_dim", "weight_decay", "bidirectional"): continue
        parts.append(f"{k}={v}")
    return "_".join(parts)

def sweep(model_type, space, device):
    defaults = {"mlp": MLPConfig(), "gru": GRUConfig(), "tf": TFConfig()}[model_type]
    keys, vals = list(space.keys()), list(space.values())
    configs = list(itertools.product(*vals))
    print(f"\n[sweep] {model_type}: {len(configs)} configs")

    done_names = _done_names()
    for i, combo in enumerate(configs):
        params = dict(zip(keys, combo))
        cfg_dict = defaults.__dict__.copy()
        cfg_dict.update(params)
        cfg = type(defaults)(**cfg_dict)
        name = _cfg_name(model_type, cfg)

        if name in done_names:
            print(f"\n--- [{i+1}/{len(configs)}] SKIP {name} (already done) ---")
            continue

        print(f"\n--- [{i+1}/{len(configs)}] ---")
        train_one(model_type, cfg, device)

    _print_top(model_type)

def _append_csv(row):
    cols = ["name","best_val_rankic","best_epoch","total_epochs","time_s"]
    df_row = pd.DataFrame([{k: row.get(k, None) for k in cols}])
    if os.path.exists(RESULTS_CSV):
        df = pd.concat([pd.read_csv(RESULTS_CSV), df_row], ignore_index=True)
    else:
        df = df_row
    df.to_csv(RESULTS_CSV, index=False)

def _done_names():
    if not os.path.exists(RESULTS_CSV): return set()
    return set(pd.read_csv(RESULTS_CSV)["name"].tolist())

def _print_top(mtype):
    if not os.path.exists(RESULTS_CSV): return
    df = pd.read_csv(RESULTS_CSV)
    df_m = df[df["name"].str.startswith(mtype)]
    print(f"\n  Top {mtype}:")
    for _, r in df_m.nlargest(5, "best_val_rankic").iterrows():
        print(f"  {r['name']:50s} IC={r['best_val_rankic']:.4f}")

# ── Main ──

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, choices=["mlp","gru","tf"])
    parser.add_argument("--sweep", nargs="?", const="all", default="all")
    parser.add_argument("--best", action="store_true")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[train] device={device}, model={args.model}")

    if args.sweep:
        if args.model == "mlp":
            sweep("mlp", MLP_SWEEP, device)
        elif args.model == "gru":
            sweep("gru", GRU_SWEEP, device)
        elif args.model == "tf":
            space = TF_SWEEP_COARSE if args.sweep in ("phase1","all","") else TF_SWEEP_FINE
            sweep("tf", space, device)
    elif args.best:
        # Use defaults (best from V1 for GRU)
        cfg = {"mlp": MLPConfig(), "gru": GRUConfig(), "tf": TFConfig()}[args.model]
        train_one(args.model, cfg, device)
    else:
        cfg = {"mlp": MLPConfig(), "gru": GRUConfig(), "tf": TFConfig()}[args.model]
        train_one(args.model, cfg, device)

if __name__ == "__main__":
    main()
