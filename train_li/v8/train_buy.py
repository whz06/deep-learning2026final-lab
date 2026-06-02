"""v8/train_buy.py — Train buy model with 3-loss comparison.

Loss options:
  listmle    — original ListMLE (baseline)
  weighted   — position-weighted ListMLE (alpha=0.9 decay)
  lambdarank — LambdaRank with NDCG gain weighting

Usage:
  python v8/train_buy.py --loss listmle
  python v8/train_buy.py --loss weighted
  python v8/train_buy.py --loss lambdarank
"""
import os, sys, time, gc, csv, argparse, numpy as np
import torch, torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF",
    "max_split_size_mb:128,roundup_power2_divisions:16,garbage_collection_threshold:0.6")

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, ROOT)

from v8.dataset_v8 import DailyStockDatasetV8
from v8.models.gru_spatial_v8 import GRURankerSpatialV8

INPUT_DIM = 31; WINDOW_SIZE = 60; EPOCHS = 100
PATIENCE = 10; GRAD_CLIP = 1.0
SCHEDULER_FACTOR = 0.5; SCHEDULER_PATIENCE = 5
WEIGHTED_ALPHA = 0.9

CKPT_DIR = os.path.join(SCRIPT_DIR, "checkpoints")
RESULTS_DIR = os.path.join(SCRIPT_DIR, "results")
RESULTS_CSV = os.path.join(RESULTS_DIR, "results_buy.csv")
os.makedirs(CKPT_DIR, exist_ok=True); os.makedirs(RESULTS_DIR, exist_ok=True)


# ===== Loss Functions =====

def listmle_loss(scores, labels):
    _, idx = torch.sort(labels, descending=True)
    s = scores[idx]
    log_cumsum = torch.logcumsumexp(s.flip(0), dim=0).flip(0)
    return -(s - log_cumsum)[:-1].sum()


def weighted_listmle_loss(scores, labels, alpha=0.9):
    _, idx = torch.sort(labels, descending=True)
    s = scores[idx]
    N = len(s)
    log_cumsum = torch.logcumsumexp(s.flip(0), dim=0).flip(0)
    w = alpha ** torch.arange(N - 1, device=s.device)
    return -((s - log_cumsum)[:-1] * w).sum() / w.sum()


def lambdarank_loss(scores, labels):
    N = len(scores)
    _, idx = torch.sort(labels, descending=True)
    y = labels[idx]
    s = scores[idx]

    gains = torch.clamp(2.0 ** (y - y.min()) - 1.0, min=0)
    discounts = 1.0 / torch.log2(torch.arange(N, device=s.device).float() + 2.)
    dcg_gain = gains * discounts

    s_diff = s.unsqueeze(1) - s.unsqueeze(0)  # [N, N]
    p_ij = torch.sigmoid(-s_diff)
    delta_ndcg = torch.abs(dcg_gain.unsqueeze(1) - dcg_gain.unsqueeze(0))

    total = (p_ij * delta_ndcg).sum() / 2  # each pair counted twice
    return total / (N * (N - 1) / 2)


LOSS_FNS = {
    "listmle": listmle_loss,
    "weighted": weighted_listmle_loss,
    "lambdarank": lambdarank_loss,
}


# ===== Metrics =====

@torch.no_grad()
def rankic(scores, labels):
    def trank(x): return torch.argsort(torch.argsort(x)).float()
    rs, rl = trank(scores), trank(labels)
    rs, rl = rs - rs.mean(), rl - rl.mean()
    denom = rs.norm() * rl.norm()
    return torch.tensor(0.0, device=scores.device) if denom < 1e-12 else (rs * rl).sum() / denom


# ===== Training =====

def train_one(args, device):
    n_ind = args.n_industries
    loss_fn = LOSS_FNS[args.loss]
    loss_name = args.loss

    tag = f"v8_{loss_name}_d{args.d_proj}_K{args.K}_H{args.hidden}_L{args.layers}_D{args.dropout}_lr{args.lr}_N{args.n_sample}_ind{n_ind}"
    ckpt_path = os.path.join(CKPT_DIR, f"{tag}.pt")

    print(f"\n{'='*60}")
    print(f" v8 Buy {loss_name}: {tag}")
    print(f"  d_proj={args.d_proj} K={args.K} n_sample={args.n_sample} n_industries={n_ind}")
    print(f"{'='*60}")

    print("[v8-buy] Loading dataset ...")
    train_ds = DailyStockDatasetV8("train", WINDOW_SIZE, sample_size=args.n_sample, shuffle=True)
    val_ds = DailyStockDatasetV8("val", WINDOW_SIZE, sample_size=None, shuffle=False)
    train_loader = DataLoader(train_ds, 1, shuffle=True, num_workers=0, pin_memory=True,
                              collate_fn=lambda x: x[0])
    val_loader = DataLoader(val_ds, 1, shuffle=False, num_workers=0, pin_memory=True,
                            collate_fn=lambda x: x[0])

    model = GRURankerSpatialV8(
        input_dim=INPUT_DIM, hidden_size=args.hidden, num_layers=args.layers,
        dropout=args.dropout, d_proj=args.d_proj, K=args.K,
        n_industries=n_ind, ind_emb_dim=args.ind_emb_dim, lambda_gate=args.lambda_gate,
    ).to(device)

    params = sum(p.numel() for p in model.parameters())
    print(f"[v8-buy] Params: {params:,}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=SCHEDULER_FACTOR, patience=SCHEDULER_PATIENCE)
    scaler = torch.amp.GradScaler("cuda", enabled=(device.type == "cuda"))

    best_ic, best_ep, no_imp = -float("inf"), 0, 0
    last_val_ic = 0.0; t0 = time.time(); epoch_times = []

    for epoch in range(1, EPOCHS + 1):
        ep_t0 = time.time()
        model.train()
        loss_sum, n_batch = 0.0, 0
        for feat, lab, ind_id in train_loader:
            feat = feat.to(device, non_blocking=True)
            lab = lab.to(device, non_blocking=True)
            ind_id = ind_id.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=(device.type == "cuda")):
                scores = model(feat, ind_id)
                loss = loss_fn(scores, lab)
            if torch.isnan(loss) or torch.isinf(loss): continue
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            scaler.step(optimizer); scaler.update()
            loss_sum += loss.item(); n_batch += 1
            if n_batch % 200 == 0: torch.cuda.empty_cache()

        should_val = (epoch % args.val_every == 0)
        if should_val:
            model.eval(); torch.cuda.empty_cache()
            val_ics = []
            with torch.no_grad():
                for feat, lab, ind_id in val_loader:
                    feat = feat.to(device, non_blocking=True)
                    lab = lab.to(device, non_blocking=True)
                    ind_id = ind_id.to(device, non_blocking=True)
                    with torch.amp.autocast("cuda", enabled=(device.type == "cuda")):
                        val_ics.append(rankic(model(feat, ind_id), lab).item())
            torch.cuda.empty_cache()
            avg_ic = float(np.mean(val_ics)) if val_ics else 0.0
        else:
            avg_ic = last_val_ic

        scheduler.step(avg_ic); last_val_ic = avg_ic

        if should_val and avg_ic > best_ic:
            best_ic, best_ep, no_imp = avg_ic, epoch, 0
            torch.save(model.state_dict(), ckpt_path)
        elif should_val:
            no_imp += 1

        ep_time = time.time() - ep_t0; epoch_times.append(ep_time)
        vflag = "V" if should_val else "T"
        if epoch <= 3 or epoch % 10 == 0 or (should_val and no_imp <= 1):
            print(f"  ep {epoch:3d} {vflag} | loss {loss_sum/max(n_batch,1):.4f} | val_ic {avg_ic:.4f} | best {best_ic:.4f} @{best_ep} | lr {optimizer.param_groups[0]['lr']:.2e} | {ep_time:.0f}s", flush=True)

        if no_imp >= PATIENCE:
            print(f"  [v8-buy] Early stop @ epoch {epoch}"); break

    elapsed = time.time() - t0; avg_ep = np.mean(epoch_times) if epoch_times else 0.0

    result = {
        "loss_fn": loss_name, "name": tag, "input_dim": INPUT_DIM,
        "d_proj": args.d_proj, "K": args.K, "hidden": args.hidden,
        "layers": args.layers, "dropout": args.dropout, "lr": args.lr,
        "n_sample": args.n_sample, "n_industries": n_ind,
        "ind_emb_dim": args.ind_emb_dim, "lambda_gate": args.lambda_gate,
        "best_val_rankic": round(best_ic, 6), "best_epoch": best_ep,
        "total_epochs": epoch, "time_min": round(elapsed/60, 1),
        "avg_ep_s": round(avg_ep, 1), "params": params,
    }
    cols = list(result.keys())
    fe = os.path.exists(RESULTS_CSV)
    with open(RESULTS_CSV, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        if not fe: w.writeheader()
        w.writerow(result)

    print(f"\n[v8-buy] DONE {loss_name}: IC={best_ic:.4f} @ep{best_ep} ({elapsed/60:.0f}min)")
    del model, optimizer, scheduler, scaler, train_loader, val_loader
    gc.collect(); torch.cuda.empty_cache()
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--loss", required=True, choices=["listmle","weighted","lambdarank"])
    parser.add_argument("--hidden", type=int, default=128)
    parser.add_argument("--layers", type=int, default=1)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--n_sample", type=int, default=1024)
    parser.add_argument("--d_proj", type=int, default=32)
    parser.add_argument("--K", type=int, default=5)
    parser.add_argument("--ind_emb_dim", type=int, default=8)
    parser.add_argument("--lambda_gate", type=float, default=0.1)
    parser.add_argument("--n_industries", type=int, default=100)
    parser.add_argument("--val-every", type=int, default=2)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"[v8-buy] device={device}")
    train_one(args, device)


if __name__ == "__main__":
    main()
