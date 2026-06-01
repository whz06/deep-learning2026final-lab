"""
v3/train_D.py — Multi-task GRU: ranking head + safety head.

Architecture:
  GRU backbone (shared) → 
    ├─ head_rank: score [ListMLE]
    └─ head_safe: sigmoid(safety>0) [BCE]

Safety label = (next_day_pct_chg > -1.0)

Joint loss: L = L_listmle + alpha × L_bce

Usage:
  python train_D.py --sweep   (sweep alpha × lr)
  python train_D.py --best    (train with best params)
"""
import os, sys, gc, time, itertools, json
import numpy as np
import pandas as pd
import torch, torch.nn as nn
from torch.utils.data import DataLoader

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF",
                      "max_split_size_mb:128,roundup_power2_divisions:16,garbage_collection_threshold:0.6")

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(SCRIPT_DIR)
V2_DIR = os.path.join(ROOT, "v2")
sys.path.insert(0, V2_DIR)
from dataset import DailyStockDataset

WINDOW_SIZE = 60
INPUT_DIM = 22
EPOCHS, PATIENCE = 50, 10
GRAD_CLIP = 1.0


class MultiTaskGRU(nn.Module):
    def __init__(self, hidden_size=128, num_layers=1, dropout=0.2):
        super().__init__()
        self.gru = nn.GRU(INPUT_DIM, hidden_size, num_layers, batch_first=True,
                          dropout=dropout if num_layers>1 else 0.0)
        # Shared head layers
        self.shared = nn.Sequential(
            nn.Linear(hidden_size, hidden_size//2), nn.ReLU(), nn.Dropout(dropout))
        # Ranking head
        self.head_rank = nn.Linear(hidden_size//2, 1)
        # Safety head
        self.head_safe = nn.Sequential(
            nn.Linear(hidden_size//2, hidden_size//4), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden_size//4, 1))

    def forward(self, x):
        out, _ = self.gru(x)
        h = self.shared(out[:, -1, :])
        score = self.head_rank(h).squeeze(-1)
        safe_logit = self.head_safe(h).squeeze(-1)
        return score, safe_logit


def listmle_loss(scores, labels):
    _, idx = torch.sort(labels, descending=True)
    s = scores[idx]
    log_cumsum = torch.logcumsumexp(s.flip(0), dim=0).flip(0)
    return -(s - log_cumsum)[:-1].sum()


@torch.no_grad()
def rankic(scores, labels):
    def trank(x): return torch.argsort(torch.argsort(x)).float()
    rs, rl = trank(scores), trank(labels)
    rs, rl = rs-rs.mean(), rl-rl.mean()
    d = rs.norm()*rl.norm()
    return torch.tensor(0.0) if d<1e-12 else (rs*rl).sum()/d


def train_one(alpha, lr, device, save_name=None):
    print(f"\nTraining: alpha={alpha}, lr={lr}")
    train_ds = DailyStockDataset("train", WINDOW_SIZE, sample_size=2048, shuffle=True)
    val_ds   = DailyStockDataset("val",   WINDOW_SIZE, sample_size=None, shuffle=False)
    train_loader = DataLoader(train_ds, 1, shuffle=True, num_workers=0, pin_memory=True, collate_fn=lambda x:x[0])
    val_loader   = DataLoader(val_ds,   1, shuffle=False, num_workers=0, pin_memory=True, collate_fn=lambda x:x[0])

    model = MultiTaskGRU().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=5)
    bce = nn.BCEWithLogitsLoss()

    best_ic, best_ep, no_imp = -float("inf"), 0, 0
    t0 = time.time()

    for epoch in range(1, EPOCHS+1):
        model.train()
        train_loss = 0.0; nb = 0
        for _, (feat, lab) in enumerate(train_loader):
            feat, lab = feat.to(device), lab.to(device)
            scores, safe_logit = model(feat)
            l_rank = listmle_loss(scores, lab)
            safe_label = (lab > -1.0).float()
            l_bce = bce(safe_logit, safe_label)
            loss = l_rank + alpha * l_bce

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            optimizer.step()
            train_loss += loss.item(); nb += 1
            if nb % 100 == 0: torch.cuda.empty_cache()

        model.eval(); torch.cuda.empty_cache()
        val_ics = []
        with torch.no_grad():
            for feat, lab in val_loader:
                feat, lab = feat.to(device), lab.to(device)
                scores, _ = model(feat)
                val_ics.append(rankic(scores, lab).item())
        torch.cuda.empty_cache()

        avg_ic = float(np.mean(val_ics))
        scheduler.step(avg_ic)
        if avg_ic > best_ic:
            best_ic, best_ep, no_imp = avg_ic, epoch, 0
            if save_name:
                torch.save(model.state_dict(), os.path.join(SCRIPT_DIR, "checkpoints", save_name))
        else:
            no_imp += 1
        if epoch%5==0 or epoch==1:
            print(f"  ep{epoch:3d} loss{train_loss/max(nb,1):.4f} "
                  f"val_ic{avg_ic:.4f} best{best_ic:.4f}@{best_ep}")
        if no_imp >= PATIENCE:
            print(f"  Early stop @{epoch}"); break

    elapsed = time.time()-t0
    print(f"  -> IC={best_ic:.4f} ({elapsed:.0f}s)")
    del model, optimizer, scheduler, train_loader, val_loader
    gc.collect(); torch.cuda.empty_cache()
    return best_ic


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--sweep", action="store_true")
    parser.add_argument("--best", action="store_true")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    os.makedirs(os.path.join(SCRIPT_DIR, "checkpoints"), exist_ok=True)

    if args.sweep:
        alphas = [0.1, 0.3, 0.5, 1.0]
        lrs = [3e-4, 5e-4, 1e-3]
        results = []
        for alpha in alphas:
            for lr in lrs:
                ic = train_one(alpha, lr, device)
                results.append({"alpha": alpha, "lr": lr, "ic": ic})

        df = pd.DataFrame(results)
        df.to_csv(os.path.join(SCRIPT_DIR, "d_sweep.csv"), index=False)
        print("\nD Sweep Results:")
        print(df.to_string(index=False))
        best = df.loc[df["ic"].idxmax()]
        print(f"\nBest: alpha={best['alpha']}, lr={best['lr']}, IC={best['ic']:.4f}")

    elif args.best:
        # Train with best params (read from sweep or use default)
        sweep_csv = os.path.join(SCRIPT_DIR, "d_sweep.csv")
        if os.path.exists(sweep_csv):
            df = pd.read_csv(sweep_csv)
            best = df.loc[df["ic"].idxmax()]
            alpha, lr = best["alpha"], best["lr"]
        else:
            alpha, lr = 0.3, 5e-4
        train_one(alpha, lr, device, save_name="multitask_best.pt")


if __name__ == "__main__":
    main()
