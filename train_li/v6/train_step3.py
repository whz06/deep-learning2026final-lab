"""v6/train_step3.py — Train GRU + SparseSpatialAttention (Step 3).

Supports --use-attn-pool flag to combine temporal attention pooling with spatial attention.
If Step 4 improves IC, use --use-attn-pool; otherwise omit to compare standalone.
"""
import os, sys, time, gc, csv, argparse
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF",
                      "max_split_size_mb:128,roundup_power2_divisions:16,garbage_collection_threshold:0.6")

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(SCRIPT_DIR)

# v2 imports (loss, metric)
V2_DIR = os.path.join(ROOT, "v2")
sys.path.insert(0, V2_DIR)
from train import listmle_loss, rankic

# v5 dataset
import importlib.util
_spec = importlib.util.spec_from_file_location(
    "v5_dataset", os.path.join(ROOT, "v5", "v5_dataset.py"))
_v5ds = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_v5ds)
DailyStockDataset = _v5ds.DailyStockDataset

# v6 model
sys.path.insert(0, ROOT)
from v6.models.gru_spatial import GRURankerSpatial

WINDOW_SIZE = 60
EPOCHS = 50
PATIENCE = 7
GRAD_CLIP = 1.0
SCHEDULER_FACTOR = 0.5
SCHEDULER_PATIENCE = 5

CKPT_DIR = os.path.join(SCRIPT_DIR, "checkpoints")
RESULTS_DIR = os.path.join(SCRIPT_DIR, "results")
RESULTS_CSV = os.path.join(RESULTS_DIR, "results.csv")
os.makedirs(CKPT_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)


def train_one(args, device):
    attn_tag = "attnpool_" if args.use_attn_pool else ""
    name = f"gru_{attn_tag}spatial_K{args.K}_H{args.hidden}_L{args.layers}_D{args.dropout}_lr{args.lr}_N{args.n_sample}"
    ckpt_path = os.path.join(CKPT_DIR, f"{name}.pt")

    print(f"\n{'='*60}")
    print(f" Step 3: GRU + SpatialAttn (K={args.K}, attn_pool={args.use_attn_pool}): {name}")
    print(f"  input_dim={args.input_dim} n_sample={args.n_sample}  "
          f"amp={not args.no_amp} compile={not args.no_compile} val_every={args.val_every}")
    print(f"{'='*60}")

    print("[step3] Loading dataset ...")
    train_ds = DailyStockDataset("train", WINDOW_SIZE, sample_size=args.n_sample, shuffle=True)
    val_ds = DailyStockDataset("val", WINDOW_SIZE, sample_size=None, shuffle=False)

    train_loader = DataLoader(train_ds, 1, shuffle=True, num_workers=0,
                              pin_memory=True, collate_fn=lambda x: x[0])
    val_loader = DataLoader(val_ds, 1, shuffle=False, num_workers=0,
                            pin_memory=True, collate_fn=lambda x: x[0])

    model = GRURankerSpatial(
        input_dim=args.input_dim, hidden_size=args.hidden,
        num_layers=args.layers, dropout=args.dropout,
        K=args.K, use_attn_pool=args.use_attn_pool,
    ).to(device)

    if not args.no_compile:
        try:
            model = torch.compile(model, mode="reduce-overhead")
            print("[step3] torch.compile: enabled (reduce-overhead)")
        except Exception as e:
            print(f"[step3] torch.compile failed: {e} — falling back to eager")

    params = sum(p.numel() for p in model.parameters())
    print(f"[step3] Parameters: {params:,}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=SCHEDULER_FACTOR, patience=SCHEDULER_PATIENCE)
    scaler = torch.amp.GradScaler("cuda", enabled=not args.no_amp and device.type == "cuda")

    best_ic, best_ep, no_imp = -float("inf"), 0, 0
    last_val_ic = 0.0
    t0 = time.time()
    epoch_times = []

    for epoch in range(1, EPOCHS + 1):
        ep_t0 = time.time()

        model.train()
        loss_sum, n_batch = 0.0, 0
        for feat, lab in train_loader:
            feat = feat.to(device, non_blocking=True)
            lab = lab.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)

            with torch.amp.autocast("cuda", enabled=not args.no_amp and device.type == "cuda"):
                scores = model(feat)
                loss = listmle_loss(scores, lab)

            if torch.isnan(loss) or torch.isinf(loss):
                print(f"  [WARN] NaN loss at batch {n_batch}, skipping", flush=True)
                continue

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            scaler.step(optimizer)
            scaler.update()

            loss_sum += loss.item()
            n_batch += 1
            if n_batch % 200 == 0:
                torch.cuda.empty_cache()

        should_val = (epoch % args.val_every == 0)

        if should_val:
            model.eval()
            torch.cuda.empty_cache()
            val_ics = []
            with torch.no_grad():
                for feat, lab in val_loader:
                    feat = feat.to(device, non_blocking=True)
                    lab = lab.to(device, non_blocking=True)
                    with torch.amp.autocast("cuda", enabled=not args.no_amp and device.type == "cuda"):
                        ic = rankic(model(feat), lab)
                    val_ics.append(ic.item())
            torch.cuda.empty_cache()
            avg_ic = float(np.mean(val_ics)) if val_ics else 0.0
        else:
            avg_ic = last_val_ic

        scheduler.step(avg_ic)
        last_val_ic = avg_ic

        if should_val and avg_ic > best_ic:
            best_ic, best_ep, no_imp = avg_ic, epoch, 0
            torch.save(model.state_dict(), ckpt_path)
        elif should_val:
            no_imp += 1

        ep_time = time.time() - ep_t0
        epoch_times.append(ep_time)

        if epoch <= 3 or epoch % 10 == 0 or (should_val and no_imp == 0):
            tag = "V" if should_val else "T"
            print(f"  ep {epoch:3d}{tag} | loss {loss_sum/max(n_batch,1):.4f} | "
                  f"val_ic {avg_ic:.4f} | best {best_ic:.4f} @{best_ep} | "
                  f"lr {optimizer.param_groups[0]['lr']:.2e} | {ep_time:.0f}s",
                  flush=True)

        if no_imp >= PATIENCE:
            print(f"  [step3] Early stop @ epoch {epoch}")
            break

    elapsed = time.time() - t0
    avg_ep = np.mean(epoch_times) if epoch_times else 0

    result = {
        "name": name, "input_dim": args.input_dim,
        "hidden": args.hidden, "layers": args.layers, "dropout": args.dropout,
        "lr": args.lr, "n_sample": args.n_sample, "K_spatial": args.K,
        "use_attn_pool": args.use_attn_pool,
        "amp": not args.no_amp, "compile": not args.no_compile,
        "best_val_rankic": round(best_ic, 6), "best_epoch": best_ep,
        "total_epochs": epoch, "time_min": round(elapsed/60, 1),
        "avg_ep_s": round(avg_ep, 1), "params": params,
    }

    cols = list(result.keys())
    fe = os.path.exists(RESULTS_CSV)
    with open(RESULTS_CSV, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        if not fe:
            w.writeheader()
        w.writerow(result)

    print(f"\n[step3] DONE: {name} -> IC={best_ic:.4f} @ ep{best_ep} ({elapsed/60:.0f}min)")
    print(f"[step3] Checkpoint -> {ckpt_path}")

    del model, optimizer, scheduler, scaler, train_loader, val_loader
    gc.collect()
    torch.cuda.empty_cache()
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dim", type=int, default=26)
    parser.add_argument("--hidden", type=int, default=128)
    parser.add_argument("--layers", type=int, default=1)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--n_sample", type=int, default=2048)
    parser.add_argument("--K", type=int, default=10, help="Number of spatial neighbors")
    parser.add_argument("--use-attn-pool", action="store_true",
                        help="Use temporal attention pooling before spatial attention")
    parser.add_argument("--val-every", type=int, default=2)
    parser.add_argument("--no-amp", action="store_true")
    parser.add_argument("--no-compile", action="store_true")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"[step3] device={device}")
    train_one(args, device)


if __name__ == "__main__":
    main()
