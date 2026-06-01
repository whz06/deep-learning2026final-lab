"""
v3/strategy_D.py — Evaluate multi-task model vs baseline on walk-forward.

Same baseline strategy (n=20,k=2), different model.
"""
import os, sys, gc, numpy as np, pandas as pd, torch
from torch.utils.data import DataLoader

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(SCRIPT_DIR)
V2_DIR = os.path.join(ROOT, "v2")
sys.path.insert(0, V2_DIR)
from dataset import DailyStockDataset
sys.path.insert(0, SCRIPT_DIR)
from train_D import MultiTaskGRU

WINDOW_SIZE = 60
TRADE_COST = 0.0013
LIMIT_UP = 9.9


def run_walkforward(ckpt_name, device):
    """Simple walk-forward: predict → buy top-20 → track returns."""
    import gc
    model = MultiTaskGRU().to(device)
    ckpt = os.path.join(SCRIPT_DIR, "checkpoints", ckpt_name)
    model.load_state_dict(torch.load(ckpt, map_location=device, weights_only=True))
    model.eval()

    val_ds = DailyStockDataset("val", WINDOW_SIZE, sample_size=None, shuffle=False)
    val_loader = DataLoader(val_ds, 1, shuffle=False, num_workers=0, pin_memory=True, collate_fn=lambda x:x[0])
    ts_map = {}
    for path in val_ds.file_paths:
        data = torch.load(path, map_location="cpu", weights_only=True)
        ts_map[path] = data.get("ts_codes", [])

    holdings, daily_rets, turnover = [], [], 0
    with torch.no_grad():
        for d_idx, (feat, lab) in enumerate(val_loader):
            feat = feat.to(device)
            scores, _ = model(feat)      # use only ranking head
            scores = scores.cpu().numpy()
            labels = lab.numpy()
            path = val_ds.file_paths[d_idx]
            codes = ts_map.get(path, [])
            pct_now = feat[:,-1,6].cpu().numpy()
            limit_up = pct_now >= LIMIT_UP

            if not holdings:
                elig = np.where(~limit_up)[0]
                if len(elig)>=20: holdings = [codes[i] for i in elig[np.argsort(scores[elig])[-20:]]]
            else:
                hs = {c:scores[codes.index(c)] for c in holdings if c in codes}
                if len(hs)>=2:
                    for c in sorted(hs,key=hs.get)[:2]: holdings.remove(c); turnover+=1
                ch = set(holdings)
                cand = [(i,s) for i,(c,s) in enumerate(zip(codes,scores)) if c not in ch and not limit_up[i]]
                cand.sort(key=lambda x:x[1],reverse=True)
                for i,_ in cand[:2]: holdings.append(codes[i]); turnover+=1
                holdings=holdings[-20:]

            hr = [labels[codes.index(c)] for c in holdings if c in codes]
            if hr:
                dr = float(np.mean(hr))
                if d_idx>0 and turnover>0: dr -= turnover*TRADE_COST/max(len(holdings),1)
                daily_rets.append(dr)
            turnover=0

    dr = np.array(daily_rets)
    cum = float(np.prod(1+dr/100)-1)*100
    sharpe = float(np.mean(dr)/(np.std(dr)+1e-8)*np.sqrt(242))
    max_dd = float(np.min(np.cumprod(1+dr/100)/np.maximum.accumulate(np.cumprod(1+dr/100))-1))*100
    ic_val = np.mean([np.corrcoef(s,l)[0,1] for _ in [0]])  # placeholder
    print(f"[strategy_D] Cum: {cum:.2f}% | Sharpe: {sharpe:.2f} | MaxDD: {max_dd:.2f}%")
    return cum, sharpe, max_dd


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--ckpt", default="multitask_best.pt")
    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    run_walkforward(args.ckpt, device)
