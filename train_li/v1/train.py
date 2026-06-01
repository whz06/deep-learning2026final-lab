"""
v1/train.py — Training loop, ListMLE loss, hyperparameter sweep, backtest.

Usage:
  python train.py                         # single run with defaults
  python train.py --sweep phase1          # coarse sweep: T × hidden
  python train.py --sweep phase2          # fine sweep: dropout × lr
  python train.py --sweep phase3          # refine: num_layers
  python train.py --best                  # train with best params, then backtest
  python train.py --device cuda           # force GPU
"""
import os
import sys
import gc
import time
import json
import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

# ── CUDA memory: limit fragmentation on Windows WDDM ──
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF",
                      "max_split_size_mb:128,roundup_power2_divisions:16,garbage_collection_threshold:0.6")

from config import (
    DEFAULT_EXP, ExpConfig, DataConfig, ModelConfig, TrainConfig,
    FEATURE_COLS, TRAIN_START, TRAIN_END, VAL_START, VAL_END,
    PHASE1_SPACE, PHASE2_SPACE, PHASE3_SPACE,
    build_sweep_configs,
)
from dataset import DailyStockDataset
from model import GRURanker

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(SCRIPT_DIR)
TRAIN_DIR = os.path.join(ROOT, "processed", "windows", "train")
VAL_DIR = os.path.join(ROOT, "processed", "windows", "val")
CKPT_DIR = os.path.join(SCRIPT_DIR, "checkpoints")
RESULTS_CSV = os.path.join(SCRIPT_DIR, "results.csv")
BEST_JSON = os.path.join(SCRIPT_DIR, "best_params.json")

os.makedirs(CKPT_DIR, exist_ok=True)


# ====================================================================
# ListMLE loss — vectorised, O(N log N)
# ====================================================================

def listmle_loss(scores: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """
    scores, labels: [N]
    Returns scalar negative log-likelihood of the correct ranking under Plackett-Luce.
    """
    _, idx = torch.sort(labels, descending=True)
    s = scores[idx]                                 # [N]
    log_cumsum = torch.logcumsumexp(s.flip(0), dim=0).flip(0)  # [N]
    ll = (s - log_cumsum)[:-1].sum()               # sum over 0 .. N-2
    return -ll


# ====================================================================
# RankIC —  pure-PyTorch Spearman
# ====================================================================

@torch.no_grad()
def rankic(scores: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """Spearman correlation coefficient (RankIC)."""
    def torch_rank(x):
        return torch.argsort(torch.argsort(x)).float()

    r_s = torch_rank(scores)
    r_l = torch_rank(labels)
    r_s = r_s - r_s.mean()
    r_l = r_l - r_l.mean()
    denom = r_s.norm() * r_l.norm()
    if denom < 1e-12:
        return torch.tensor(0.0, device=scores.device)
    return (r_s * r_l).sum() / denom


# ====================================================================
# Single training run
# ====================================================================

def train_one_config(cfg: ExpConfig, device: torch.device, verbose: bool = True,
                     save_best: bool = True) -> dict:
    """Train one ExpConfig. Returns dict of metrics."""

    if verbose:
        print(f"\n{'='*60}")
        print(f"Training: {cfg.name}")
        print(f"  data:   T={cfg.data.window_size}, batch={cfg.data.batch_size}")
        print(f"  model:  H={cfg.model.hidden_size}, L={cfg.model.num_layers}, "
              f"D={cfg.model.dropout}, Bi={cfg.model.bidirectional}")
        print(f"  train:  lr={cfg.train.lr}, wd={cfg.train.weight_decay}, "
              f"epochs={cfg.train.epochs}, patience={cfg.train.patience}")
        print(f"{'='*60}")

    # Build dataset — use the correct window_size sub-directory if exists
    train_dir_actual = TRAIN_DIR
    val_dir_actual = VAL_DIR

    train_ds = DailyStockDataset(train_dir_actual, window_size=cfg.data.window_size,
                                  sample_size=cfg.data.batch_size, shuffle=True)
    val_ds = DailyStockDataset(val_dir_actual, window_size=cfg.data.window_size,
                                sample_size=None, shuffle=False)

    train_loader = DataLoader(train_ds, batch_size=1, shuffle=True, num_workers=0,
                               pin_memory=True, collate_fn=lambda x: x[0])
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False, num_workers=0,
                             pin_memory=True, collate_fn=lambda x: x[0])

    model = GRURanker(
        input_dim=cfg.model.input_dim,
        hidden_size=cfg.model.hidden_size,
        num_layers=cfg.model.num_layers,
        dropout=cfg.model.dropout,
        bidirectional=cfg.model.bidirectional,
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.train.lr,
                                  weight_decay=cfg.train.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=cfg.train.scheduler_factor,
        patience=cfg.train.scheduler_patience,
    )

    best_ic = -float("inf")
    best_epoch = 0
    epochs_no_improve = 0
    t_start = time.time()

    for epoch in range(1, cfg.train.epochs + 1):
        model.train()
        train_loss_sum = 0.0
        train_n = 0

        for batch_i, (features, labels) in enumerate(train_loader):
            features = features.to(device)
            labels = labels.to(device)

            optimizer.zero_grad(set_to_none=True)
            scores = model(features)
            loss = listmle_loss(scores, labels)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), cfg.train.grad_clip)
            optimizer.step()

            train_loss_sum += loss.item()
            train_n += 1

            # Periodically release CUDA cache to fight WDDM fragmentation
            if batch_i % 100 == 0 and batch_i > 0:
                torch.cuda.empty_cache()

        # Validation
        model.eval()
        torch.cuda.empty_cache()
        val_ics = []
        with torch.no_grad():
            for features, labels in val_loader:
                features = features.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)
                scores = model(features)
                ic = rankic(scores, labels)
                val_ics.append(ic.item())

        torch.cuda.empty_cache()

        avg_val_ic = float(np.mean(val_ics)) if val_ics else 0.0
        avg_train_loss = train_loss_sum / max(train_n, 1)
        scheduler.step(avg_val_ic)

        if avg_val_ic > best_ic:
            best_ic = avg_val_ic
            best_epoch = epoch
            epochs_no_improve = 0
            if save_best:
                ckpt_path = os.path.join(CKPT_DIR, f"{cfg.name}.pt")
                torch.save(model.state_dict(), ckpt_path)
        else:
            epochs_no_improve += 1

        if verbose and (epoch % 5 == 0 or epoch == 1):
            print(f"  epoch {epoch:3d} | loss {avg_train_loss:.4f} | "
                  f"val_ic {avg_val_ic:.4f} | best {best_ic:.4f} @ {best_epoch} | "
                  f"lr {optimizer.param_groups[0]['lr']:.2e}")

        if epochs_no_improve >= cfg.train.patience:
            if verbose:
                print(f"  Early stopping at epoch {epoch}")
            break

    train_time = time.time() - t_start

    result = {
        "name": cfg.name,
        "window_size": cfg.data.window_size,
        "hidden_size": cfg.model.hidden_size,
        "num_layers": cfg.model.num_layers,
        "dropout": cfg.model.dropout,
        "bidirectional": cfg.model.bidirectional,
        "lr": cfg.train.lr,
        "batch_size": cfg.data.batch_size,
        "best_val_rankic": round(best_ic, 6),
        "best_epoch": best_epoch,
        "total_epochs": epoch,
        "train_time_s": round(train_time, 1),
    }

    if verbose:
        print(f"  -> best val RankIC = {best_ic:.4f} @ epoch {best_epoch} "
              f"({train_time:.0f}s)")

    del model, optimizer, scheduler, train_loader, val_loader
    gc.collect()
    torch.cuda.empty_cache()

    return result


# ====================================================================
# Sweep orchestrator
# ====================================================================

def run_sweep(phase: str, device: torch.device):
    base = DEFAULT_EXP
    if phase == "phase1":
        space = PHASE1_SPACE
        tag = "phase1"
    elif phase == "phase2":
        space = PHASE2_SPACE
        tag = "phase2"
        # Inherit best from phase1
        best_p1 = _load_best_from_csv()
        if best_p1:
            base.data.window_size = best_p1.get("window_size", base.data.window_size)
            base.model.hidden_size = best_p1.get("hidden_size", base.model.hidden_size)
            print(f"[sweep] Phase2 inherits: T={base.data.window_size}, H={base.model.hidden_size}")
    elif phase == "phase3":
        space = PHASE3_SPACE
        tag = "phase3"
        best_p2 = _load_best_from_csv()
        if best_p2:
            for k in ["window_size", "hidden_size", "dropout", "lr"]:
                if k in best_p2:
                    if k in ("window_size",):
                        setattr(base.data, k, best_p2[k])
                    elif k in ("hidden_size", "dropout"):
                        setattr(base.model, k, best_p2[k])
                    elif k == "lr":
                        base.train.lr = best_p2[k]
            print(f"[sweep] Phase3 inherits best from previous stages")
    else:
        raise ValueError(f"Unknown phase: {phase}")

    configs = build_sweep_configs(space, base)
    print(f"\n[sweep] {tag}: {len(configs)} configs to evaluate\n")

    all_results = []
    for i, cfg in enumerate(configs):
        cfg.name = f"{tag}_{cfg.name}"
        print(f"\n--- [{i + 1}/{len(configs)}] ---")
        res = train_one_config(cfg, device, verbose=True, save_best=True)
        all_results.append(res)
        _append_csv(res)

    _print_summary(all_results, tag)
    return all_results


def run_best(device: torch.device):
    """Train final model with best hyperparams, then run backtest."""
    best = _load_best_from_csv()
    cfg = DEFAULT_EXP
    cfg.name = "best"

    if best:
        for k, v in best.items():
            if k in ("window_size", "batch_size"):
                setattr(cfg.data, k, v)
            elif k in ("hidden_size", "num_layers", "dropout", "bidirectional"):
                setattr(cfg.model, k, v)
            elif k in ("lr", "weight_decay"):
                setattr(cfg.train, k, v)
        print(f"[best] Using: T={cfg.data.window_size}, H={cfg.model.hidden_size}, "
              f"L={cfg.model.num_layers}, D={cfg.model.dropout}, lr={cfg.train.lr}")
    else:
        print("[best] No prior results found. Using default hyperparams.")

    cfg.train.epochs = 50
    cfg.train.patience = 15
    res = train_one_config(cfg, device, verbose=True, save_best=True)
    _append_csv(res)

    # Backtest
    print("\n[backtest] Running backtest on validation set ...")
    ckpt_path = os.path.join(CKPT_DIR, "best.pt")
    backtest_result = run_backtest(cfg, device, ckpt_path)
    print(f"  Cumulative return: {backtest_result['cum_return']:.4%}")
    print(f"  Annualised Sharpe: {backtest_result['sharpe']:.4f}")
    print(f"  Max drawdown: {backtest_result['max_drawdown']:.4%}")

    # Save best params for the sweep orchestrator
    with open(BEST_JSON, "w") as f:
        json.dump(res, f, indent=2)

    return res


# ====================================================================
# Backtest —  T+1 rotation on validation period
# ====================================================================

def run_backtest(cfg: ExpConfig, device: torch.device, ckpt_path: str,
                 top_n: int = 20, rebalance_k: int = 2) -> dict:
    """Simple T+1 rotation backtest."""
    model = GRURanker(
        input_dim=cfg.model.input_dim, hidden_size=cfg.model.hidden_size,
        num_layers=cfg.model.num_layers, dropout=cfg.model.dropout,
        bidirectional=cfg.model.bidirectional,
    ).to(device)
    model.load_state_dict(torch.load(ckpt_path, map_location=device, weights_only=True))
    model.eval()

    val_ds = DailyStockDataset(VAL_DIR, window_size=cfg.data.window_size,
                                sample_size=None, shuffle=False)
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False, num_workers=0,
                             pin_memory=True, collate_fn=lambda x: x[0])

    holdings = []          # list of ts_code currently held
    daily_returns = []

    with torch.no_grad():
        for day_idx, (features, labels) in enumerate(val_loader):
            features = features.to(device)
            labels_np = labels.numpy()

            scores = model(features).cpu().numpy()  # [N]
            data = torch.load(val_ds.file_paths[day_idx], map_location="cpu", weights_only=True)
            ts_codes = data["ts_codes"]

            if len(holdings) == 0:
                # First day: buy top_n
                top_idx = np.argsort(scores)[-top_n:]
                holdings = [ts_codes[i] for i in top_idx]
            else:
                # Sell bottom k from holdings, buy top k from all
                held_scores = {c: scores[ts_codes.index(c)] for c in holdings if c in ts_codes}
                if len(held_scores) >= rebalance_k:
                    sell = sorted(held_scores, key=held_scores.get)[:rebalance_k]
                    for c in sell:
                        holdings.remove(c)

                current_held = set(holdings)
                candidates = [(i, s) for i, (c, s) in enumerate(zip(ts_codes, scores))
                             if c not in current_held]
                candidates.sort(key=lambda x: x[1], reverse=True)
                for i, _ in candidates[:rebalance_k]:
                    holdings.append(ts_codes[i])

                holdings = holdings[-top_n:]  # keep at most n

            # Compute portfolio return on THIS day
            held_returns = []
            for c in holdings:
                if c in ts_codes:
                    idx = ts_codes.index(c)
                    held_returns.append(labels_np[idx])
            if held_returns:
                daily_returns.append(float(np.mean(held_returns)))

    daily_returns = np.array(daily_returns)
    cum_return = float(np.prod(1 + daily_returns / 100) - 1)
    sharpe = float(np.mean(daily_returns) / (np.std(daily_returns) + 1e-8) * np.sqrt(242))
    max_dd = float(_max_drawdown(daily_returns))

    return {"cum_return": cum_return, "sharpe": sharpe, "max_drawdown": max_dd}


def _max_drawdown(returns: np.ndarray) -> float:
    """Max drawdown from daily percentage returns."""
    cum = np.cumprod(1 + returns / 100)
    peak = np.maximum.accumulate(cum)
    dd = (cum - peak) / peak
    return float(np.min(dd))


# ====================================================================
# CSV helpers
# ====================================================================

def _append_csv(row: dict):
    cols = ["name", "window_size", "hidden_size", "num_layers", "dropout",
            "bidirectional", "lr", "batch_size", "best_val_rankic",
            "best_epoch", "total_epochs", "train_time_s"]
    df_row = pd.DataFrame([{k: row.get(k, None) for k in cols}])
    if os.path.exists(RESULTS_CSV):
        df = pd.read_csv(RESULTS_CSV)
        df = pd.concat([df, df_row], ignore_index=True)
    else:
        df = df_row
    df.to_csv(RESULTS_CSV, index=False)


def _load_best_from_csv() -> dict | None:
    if not os.path.exists(RESULTS_CSV):
        return None
    df = pd.read_csv(RESULTS_CSV)
    if len(df) == 0:
        return None
    best_row = df.loc[df["best_val_rankic"].idxmax()]
    return best_row.to_dict()


def _print_summary(results: list, tag: str):
    print(f"\n{'='*60}")
    print(f"  Sweep {tag} Summary  (best_val_rankic)")
    print(f"{'='*60}")
    df = pd.DataFrame(results)
    top = df.nlargest(5, "best_val_rankic")
    for _, r in top.iterrows():
        print(f"  {r['name']:30s}  IC={r['best_val_rankic']:.4f}  "
              f"ep={r['best_epoch']}  T={r['window_size']}  "
              f"H={r['hidden_size']}  L={r['num_layers']}  "
              f"D={r['dropout']}  lr={r['lr']}")


# ====================================================================
# Main
# ====================================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sweep", type=str, default="", choices=["phase1", "phase2", "phase3", "all"])
    parser.add_argument("--best", action="store_true")
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    if args.device == "cuda" and torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    print(f"[train] device = {device}")

    if args.sweep in ("phase1", "phase2", "phase3"):
        run_sweep(args.sweep, device)
    elif args.sweep == "all":
        for p in ("phase1", "phase2", "phase3"):
            run_sweep(p, device)
    elif args.best:
        run_best(device)
    else:
        # Single default run
        cfg = DEFAULT_EXP
        cfg.name = "baseline"
        train_one_config(cfg, device, verbose=True)


if __name__ == "__main__":
    main()
    sys.exit(0)
