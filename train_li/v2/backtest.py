"""
v2/backtest.py — T+1 rotation backtest with transaction costs & price limit checks.

Cost: 0.13% per trade (印花税0.1% + 佣金0.03%)
"""
import os, sys
import numpy as np
import torch
from torch.utils.data import DataLoader

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

from config import WINDOW_SIZE
from dataset import DailyStockDataset, get_ts_codes
from models import MLPRanker, GRURanker, TransformerRanker
from config import MLPConfig, GRUConfig, TFConfig


TRADE_COST = 0.0013          # 0.13% per side
LIMIT_UP_THRESHOLD = 9.9     # ignore stocks hitting limit-up (can't buy)


def _parse_ckpt(model_type: str, ckpt_name: str):
    """Extract model config from checkpoint filename."""
    import re
    name = ckpt_name.replace(".pt", "")
    if model_type == "mlp":
        hd = int(re.search(r"hidden_dim=(\d+)", name).group(1))
        nl = int(re.search(r"n_layers=(\d+)", name).group(1))
        do = float(re.search(r"dropout=([\d.]+)_lr", name).group(1))
        return {"input_dim": MLPConfig.input_dim, "hidden_dim": hd,
                "n_layers": nl, "dropout": do}
    elif model_type == "gru":
        hs = int(re.search(r"hidden_size=(\d+)", name).group(1))
        nl = int(re.search(r"num_layers=(\d+)", name).group(1))
        do = float(re.search(r"dropout=([\d.]+)_lr", name).group(1))
        return {"input_dim": GRUConfig.input_dim, "hidden_size": hs,
                "num_layers": nl, "dropout": do}
    elif model_type == "tf":
        dm = int(re.search(r"d_model=(\d+)", name).group(1))
        nh = int(re.search(r"n_heads=(\d+)", name).group(1))
        nt = int(re.search(r"n_temporal_layers=(\d+)", name).group(1))
        ns = int(re.search(r"n_spatial_layers=(\d+)", name).group(1))
        do = float(re.search(r"dropout=([\d.]+)_lr", name).group(1))
        return {"input_dim": TFConfig.input_dim, "d_model": dm, "n_heads": nh,
                "n_temporal_layers": nt, "n_spatial_layers": ns, "dropout": do}
    return {}

def run_backtest(model_type: str, ckpt_name: str, device,
                 top_n: int = 20, rebalance_k: int = 2,
                 use_costs: bool = True):
    print(f"[backtest] {model_type} from {ckpt_name}")
    params = _parse_ckpt(model_type, ckpt_name)
    model = {"mlp": MLPRanker, "gru": GRURanker, "tf": TransformerRanker}[model_type](**params)
    ckpt_path = os.path.join(SCRIPT_DIR, "checkpoints", ckpt_name)
    model.load_state_dict(torch.load(ckpt_path, map_location=device, weights_only=True))
    model.to(device)
    model.eval()

    val_ds = DailyStockDataset("val", WINDOW_SIZE, sample_size=None, shuffle=False)
    val_loader = DataLoader(val_ds, 1, shuffle=False, num_workers=0,
                            pin_memory=True, collate_fn=lambda x: x[0])
    ts_codes_map = get_ts_codes("val")

    holdings = []
    daily_returns = []
    turnover_count = 0

    with torch.no_grad():
        for d_idx, (feat, lab) in enumerate(val_loader):
            feat = feat.to(device)
            scores = model(feat).cpu().numpy()
            labels = lab.numpy()
            path = val_ds.file_paths[d_idx]
            codes = ts_codes_map.get(path, [])

            # Limit-up filter: can't buy stocks that hit limit-up today
            # pct_chg is feature index 6 at last timestep
            pct_now = feat[:, -1, 6].cpu().numpy()
            limit_up = pct_now >= LIMIT_UP_THRESHOLD

            if len(holdings) == 0:
                eligible = np.where(~limit_up)[0]
                if len(eligible) >= top_n:
                    top_idx = eligible[np.argsort(scores[eligible])[-top_n:]]
                    holdings = [codes[i] for i in top_idx]
            else:
                # Sell lowest k from holdings
                if len(holdings) >= rebalance_k:
                    held_scores = {c: scores[codes.index(c)]
                                   for c in holdings if c in codes}
                    if len(held_scores) >= rebalance_k:
                        sell = sorted(held_scores, key=held_scores.get)[:rebalance_k]
                        for c in sell:
                            holdings.remove(c)
                            turnover_count += 1

                # Buy top k from eligible non-held stocks
                current_held = set(holdings)
                candidates = [(i, s) for i, (c, s) in enumerate(zip(codes, scores))
                              if c not in current_held and not limit_up[i]]
                candidates.sort(key=lambda x: x[1], reverse=True)
                for i, _ in candidates[:rebalance_k]:
                    holdings.append(codes[i])
                    turnover_count += 1

                holdings = holdings[-top_n:]

            # Compute portfolio return
            held_rets = []
            for c in holdings:
                if c in codes:
                    idx = codes.index(c)
                    held_rets.append(labels[idx])
            if held_rets:
                daily_return = float(np.mean(held_rets))
                if use_costs and d_idx > 0:
                    cost_ratio = (turnover_count * TRADE_COST / max(len(holdings), 1))
                    daily_return -= cost_ratio
                daily_returns.append(daily_return)
            turnover_count = 0

    daily_returns = np.array(daily_returns)
    cum_return = float(np.prod(1 + daily_returns / 100) - 1)
    sharpe = float(np.mean(daily_returns) / (np.std(daily_returns) + 1e-8) * np.sqrt(242))
    max_dd = _max_drawdown(daily_returns)

    print(f"  Cumulative return: {cum_return:.4%}")
    print(f"  Annualised Sharpe: {sharpe:.4f}")
    print(f"  Max drawdown:      {max_dd:.4%}")
    return {"cum_return": cum_return, "sharpe": sharpe, "max_drawdown": max_dd}


def _max_drawdown(returns):
    cum = np.cumprod(1 + returns / 100)
    return float(np.min(cum / np.maximum.accumulate(cum) - 1))


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, choices=["mlp","gru","tf"])
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--no-costs", action="store_true")
    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    run_backtest(args.model, args.ckpt, device, use_costs=not args.no_costs)
