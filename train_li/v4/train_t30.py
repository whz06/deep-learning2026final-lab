"""v4/train_t30.py — Train a T=30 GRU model, reusing existing v2 dataset (auto-truncates).

Reuses v2's:
  - DailyStockDataset (truncates features[:, -W:, :] at load)
  - listmle_loss, rankic
  - GRURanker architecture

Existing v2_windows .pt files (T=60) are loaded and last 30 steps taken automatically.
No need to rebuild windows.

Best GRU config from v2 sweep: H=128, L=1, D=0.2, lr=3e-4
Checkpoint saved to v4/checkpoints/gru_t30_h128_l1_d0.2.pt

Usage:
  python v4/train_t30.py              # default cuda
  python v4/train_t30.py --device cpu  # fallback
"""

import os, sys, time, gc, argparse
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF",
                      "max_split_size_mb:128,roundup_power2_divisions:16,garbage_collection_threshold:0.6")

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(SCRIPT_DIR)
V2_DIR = os.path.join(ROOT, "v2")
sys.path.insert(0, V2_DIR)

from train import listmle_loss, rankic
from dataset import DailyStockDataset
from models.gru import GRURanker

WINDOW_SIZE = 30
BATCH_SIZE = 2048
EPOCHS = 50
PATIENCE = 10
GRAD_CLIP = 1.0
SCHEDULER_FACTOR = 0.5
SCHEDULER_PATIENCE = 5

GRU_CONFIG = {
    "input_dim": 22,
    "hidden_size": 128,
    "num_layers": 1,
    "dropout": 0.2,
    "lr": 3e-4,
    "weight_decay": 1e-5,
}

CKPT_DIR = os.path.join(SCRIPT_DIR, "checkpoints")
os.makedirs(CKPT_DIR, exist_ok=True)
CKPT_NAME = "gru_t30_h128_l1_d0.2.pt"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"[train_t30] device={device}")
    print(f"[train_t30] config: H={GRU_CONFIG['hidden_size']} "
          f"L={GRU_CONFIG['num_layers']} D={GRU_CONFIG['dropout']} "
          f"lr={GRU_CONFIG['lr']} window={WINDOW_SIZE}")

    print("[train_t30] loading train set (may take 1-2 min for float16 cache) ...")
    train_ds = DailyStockDataset("train", WINDOW_SIZE, sample_size=BATCH_SIZE, shuffle=True)
    val_ds = DailyStockDataset("val", WINDOW_SIZE, sample_size=None, shuffle=False)

    train_loader = DataLoader(train_ds, batch_size=1, shuffle=True, num_workers=0,
                              pin_memory=True, collate_fn=lambda x: x[0])
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False, num_workers=0,
                            pin_memory=True, collate_fn=lambda x: x[0])

    model = GRURanker(
        GRU_CONFIG["input_dim"],
        GRU_CONFIG["hidden_size"],
        GRU_CONFIG["num_layers"],
        GRU_CONFIG["dropout"],
    ).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=GRU_CONFIG["lr"],
        weight_decay=GRU_CONFIG["weight_decay"],
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=SCHEDULER_FACTOR, patience=SCHEDULER_PATIENCE,
    )

    best_ic, best_ep, no_imp = -float("inf"), 0, 0
    ckpt_path = os.path.join(CKPT_DIR, CKPT_NAME)
    t0 = time.time()

    for epoch in range(1, EPOCHS + 1):
        model.train()
        loss_sum, n_batch = 0.0, 0
        for feat, lab in train_loader:
            feat, lab = feat.to(device, non_blocking=True), lab.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            scores = model(feat)
            loss = listmle_loss(scores, lab)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            optimizer.step()
            loss_sum += loss.item()
            n_batch += 1
            if n_batch % 100 == 0:
                torch.cuda.empty_cache()

        model.eval()
        torch.cuda.empty_cache()
        val_ics = []
        with torch.no_grad():
            for feat, lab in val_loader:
                feat, lab = feat.to(device, non_blocking=True), lab.to(device, non_blocking=True)
                ic = rankic(model(feat), lab)
                val_ics.append(ic.item())
        torch.cuda.empty_cache()

        avg_ic = float(np.mean(val_ics)) if val_ics else 0.0
        scheduler.step(avg_ic)

        if avg_ic > best_ic:
            best_ic, best_ep, no_imp = avg_ic, epoch, 0
            torch.save(model.state_dict(), ckpt_path)
        else:
            no_imp += 1

        if epoch % 5 == 0 or epoch == 1:
            print(f"  ep {epoch:3d} | loss {loss_sum / max(n_batch, 1):.4f} | "
                  f"val_ic {avg_ic:.4f} | best {best_ic:.4f} @ {best_ep} | "
                  f"lr {optimizer.param_groups[0]['lr']:.2e}")

        if no_imp >= PATIENCE:
            print(f"  [train_t30] Early stop @ epoch {epoch}")
            break

    elapsed = time.time() - t0
    print(f"\n[train_t30] DONE: best_val_ic={best_ic:.4f} @ ep{best_ep} "
          f"({elapsed:.0f}s, {epoch} epochs)")
    print(f"[train_t30] Checkpoint -> {ckpt_path}")

    del model, optimizer, scheduler, train_loader, val_loader
    gc.collect()
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
