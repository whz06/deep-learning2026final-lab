"""
v2/testing/analyze_strategy.py — Deep dive into GRU trading behavior.

Runs one segment (10 days) and logs:
  - Daily score distribution of held stocks vs all stocks
  - How often top-K scored stocks actually outperform
  - Turnover analysis
  - Score dispersion (max-min)
"""
import os, sys, numpy as np, pandas as pd, torch, gc

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
V2_DIR = os.path.dirname(SCRIPT_DIR)
ROOT = os.path.dirname(V2_DIR)
PARQUET_PATH = os.path.join(ROOT, "processed", "all_data.parquet")
INDEX_PATH   = os.path.join(ROOT, "data", "market", "000300.SH.csv")
CKPT_DIR = os.path.join(V2_DIR, "checkpoints")
sys.path.insert(0, V2_DIR)
from models.gru import GRURanker

WINDOW_SIZE = 60
RAW_FEATURES = ["open","high","low","close","vol","amount","pct_chg",
                "turnover_rate","volume_ratio","total_mv"]
TECH_FEATURES = ["macd","macd_signal","rsi","bb_width","bb_pct",
                 "mom_5","mom_20","vol_20"]
GRU_CKPT = "gru_gru_hidden_size=128_num_layers=1_dropout=0.2_lr=0.0003.pt"
TRADE_COST = 0.0013

def add_tech(df):
    close = df["close"].astype(float)
    e12,e26=close.ewm(span=12,adjust=False).mean(),close.ewm(span=26,adjust=False).mean()
    df["macd"]=e12-e26; df["macd_signal"]=df["macd"].ewm(span=9,adjust=False).mean()
    d=close.diff(); g=d.clip(lower=0); l=(-d).clip(lower=0)
    rs=g.ewm(alpha=1/14,adjust=False).mean()/(l.ewm(alpha=1/14,adjust=False).mean()+1e-8)
    df["rsi"]=100-100/(1+rs)
    m20,s20=close.rolling(20).mean(),close.rolling(20).std()
    df["bb_width"]=2*s20/(m20+1e-8); df["bb_pct"]=(close-(m20-2*s20))/(4*s20+1e-8)
    df["mom_5"]=close/close.shift(5)-1; df["mom_20"]=close/close.shift(20)-1
    df["vol_20"]=close.pct_change().rolling(20).std()
    return df

def build_window(date, stock_series, idx_map):
    feats, codes = [], []
    for ts_code, sdf in stock_series.items():
        sd = sdf[sdf["trade_date"] <= date]
        if len(sd) < WINDOW_SIZE+1: continue
        sw = sd.iloc[-WINDOW_SIZE-1:]
        rv = sw[RAW_FEATURES + TECH_FEATURES].values.astype(np.float32)
        if np.isnan(rv).any(): continue
        feats.append(rv[-WINDOW_SIZE:]); codes.append(ts_code)
    if not feats: return None, None
    fa = np.stack(feats, axis=0); n = len(feats)
    p_vals = fa[:, -1, 6]; a_vals = fa[:, -1, 5]; t_vals = fa[:, -1, 7]
    pr = np.argsort(np.argsort(p_vals)).astype(np.float32)/max(n-1,1)
    ar = np.argsort(np.argsort(a_vals)).astype(np.float32)/max(n-1,1)
    tr_ = np.argsort(np.argsort(t_vals)).astype(np.float32)/max(n-1,1)
    ip = idx_map.get(date, 0.0); rb = np.full(n, p_vals - ip, dtype=np.float32)
    cr = np.stack([np.tile(pr[:,None],(1,WINDOW_SIZE)),np.tile(ar[:,None],(1,WINDOW_SIZE)),
                    np.tile(tr_[:,None],(1,WINDOW_SIZE)),np.tile(rb[:,None],(1,WINDOW_SIZE))],-1)
    full = np.concatenate([fa, cr], -1)
    m,s=full.mean(axis=0,keepdims=True),full.std(axis=0,keepdims=True)+1e-8
    full = (full-m)/s
    return full, codes

def main():
    device = torch.device("cuda")
    # Load model
    m = GRURanker(22,128,1,0.2)
    m.load_state_dict(torch.load(os.path.join(CKPT_DIR, GRU_CKPT), map_location=device, weights_only=True))
    m.to(device).eval()

    # Load data
    df = pd.read_parquet(PARQUET_PATH); df["trade_date"] = df["trade_date"].astype(str)
    dates_all = sorted(df["trade_date"].unique())
    tc = [d for d in dates_all if d >= "20260201"]
    lb_idx = max(0, dates_all.index(tc[0]) - WINDOW_SIZE - 1 - 20)
    df = df[df["trade_date"] >= dates_all[lb_idx]]
    idx_map = dict(zip(pd.read_csv(INDEX_PATH,dtype={"trade_date":str})["trade_date"],
                       pd.read_csv(INDEX_PATH,dtype={"trade_date":str})["pct_chg"]))

    stocks = sorted(df["ts_code"].unique())
    stock_series = {}
    for ts in stocks:
        stock_series[ts] = add_tech(df[df["ts_code"]==ts].sort_values("trade_date").reset_index(drop=True))
    print(f"[data] {len(stocks)} stocks loaded")

    test_dates = [d for d in dates_all if "20260201" <= d <= "20260531"]
    seg = test_dates[:11]  # 10 days + 1 for label
    print(f"[segment] {seg[0]} → {seg[-1]}\n")

    holdings = []; turnover = 0
    for pi in range(10):
        pdate, ndate = seg[pi], seg[pi+1]
        features, codes = build_window(pdate, stock_series, idx_map)
        if features is None: continue
        scores = m(torch.from_numpy(features).to(device)).detach().cpu().numpy()
        torch.cuda.empty_cache()

        # Next-day actual returns
        vl, vi = [], []
        for ci, c in enumerate(codes):
            r = stock_series[c][stock_series[c]["trade_date"]==ndate]
            if len(r)>0: vl.append(r["pct_chg"].values[0]); vi.append(ci)
        if not vi: continue
        cv, sv, lv = [codes[i] for i in vi], scores[vi], np.array(vl)
        limit_up = features[vi,-1,6] >= 9.9

        # Rank analysis
        ranked_idx = np.argsort(sv)[::-1]
        top10_idx  = ranked_idx[:10]
        top20_idx  = ranked_idx[:20]
        bot10_idx  = ranked_idx[-10:]
        all_ret_mean = lv[~limit_up].mean() if (~limit_up).sum()>0 else 0

        print(f"[{pdate}→{ndate}]")
        print(f"  All stocks mean ret: {lv.mean():+.2f}%")
        print(f"  Top-10 mean ret:     {lv[top10_idx].mean():+.2f}%")
        print(f"  Top-20 mean ret:     {lv[top20_idx].mean():+.2f}%")
        print(f"  Bot-10 mean ret:     {lv[bot10_idx].mean():+.2f}%")
        print(f"  IC this day:         {np.corrcoef(sv,lv)[0,1]:.4f}")
        print(f"  Score range:         {sv.min():.2f} ~ {sv.max():.2f}")

        # Trading simulation: what would have happened
        if not holdings:
            elig = np.where(~limit_up)[0]
            if len(elig)>=20:
                idx = elig[np.argsort(sv[elig])[-20:]]
                holdings = [cv[i] for i in idx]
        else:
            hs = {c:sv[cv.index(c)] for c in holdings if c in cv}
            if len(hs)>=2:
                sell = sorted(hs,key=hs.get)[:2]
                for c in sell: holdings.remove(c); turnover+=1
            ch=set(holdings)
            cand=[(i,s) for i,(c,s) in enumerate(zip(cv,sv)) if c not in ch and not limit_up[i]]
            cand.sort(key=lambda x:x[1],reverse=True)
            for i,_ in cand[:2]: holdings.append(cv[i]); turnover+=1
            holdings=holdings[-20:]

        hr=[lv[cv.index(c)] for c in holdings if c in cv]
        if hr:
            dr=float(np.mean(hr))
            if pi>0 and turnover>0: dr-=turnover*TRADE_COST/max(len(holdings),1)
            held_mean=float(np.mean(hr))
            print(f"  Held({len(holdings)}): mean ret {dr:+.2f}%  "
                  f"vs all {all_ret_mean:+.2f}%")
        turnover=0

        # Score distribution of held stocks
        held_scores = [sv[cv.index(c)] for c in holdings if c in cv]
        if held_scores:
            print(f"  Held score: mean={np.mean(held_scores):.2f} "
                  f"vs all mean={sv.mean():.2f}")
        print()

    del m; gc.collect(); torch.cuda.empty_cache()

if __name__ == "__main__":
    main()
