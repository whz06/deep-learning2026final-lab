"""验证 A: 邻居GRU分数是否包含增量信息？5分钟快速验证"""
import numpy as np, pandas as pd, torch, os, sys
from scipy.stats import spearmanr
from sklearn.neighbors import NearestNeighbors

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
V2_DIR = os.path.join(ROOT, "v2")
sys.path.insert(0, V2_DIR)
from models.gru import GRURanker

W = 60
RAW = ["open","high","low","close","vol","amount","pct_chg","turnover_rate","volume_ratio","total_mv"]
TECH = ["macd","macd_signal","rsi","bb_width","bb_pct","mom_5","mom_20","vol_20"]
CKPT = "gru_gru_hidden_size=128_num_layers=1_dropout=0.2_lr=0.0003.pt"
N_STOCKS = 600

def add_tech(df):
    c = df["close"].astype(float)
    e12=c.ewm(span=12,adjust=False).mean(); e26=c.ewm(span=26,adjust=False).mean()
    df["macd"]=e12-e26; df["macd_signal"]=df["macd"].ewm(span=9,adjust=False).mean()
    d=c.diff(); g=d.clip(lower=0); l=(-d).clip(lower=0)
    rs=g.ewm(alpha=1/14,adjust=False).mean()/(l.ewm(alpha=1/14,adjust=False).mean()+1e-8)
    df["rsi"]=100-100/(1+rs)
    m20,s20=c.rolling(20).mean(),c.rolling(20).std()
    df["bb_width"]=2*s20/(m20+1e-8); df["bb_pct"]=(c-(m20-2*s20))/(4*s20+1e-8)
    df["mom_5"]=c/c.shift(5)-1; df["mom_20"]=c/c.shift(20)-1
    df["vol_20"]=c.pct_change().rolling(20).std()
    return df

def build(date, series, idx_map, W):
    feats,codes=[],[]
    for ts,sdf in series.items():
        sd=sdf[sdf["trade_date"]<=date]
        if len(sd)<W+1: continue
        sw=sd.iloc[-W-1:]; rv=sw[RAW+TECH].values.astype(np.float32)
        if np.isnan(rv).any(): continue
        feats.append(rv[-W:]); codes.append(ts)
    if not feats: return None,None,None
    fa=np.stack(feats,0); n=len(feats)
    pv,av,tv=fa[:,-1,6],fa[:,-1,5],fa[:,-1,7]
    pr=np.argsort(np.argsort(pv)).astype(np.float32)/max(n-1,1)
    ar=np.argsort(np.argsort(av)).astype(np.float32)/max(n-1,1)
    tr=np.argsort(np.argsort(tv)).astype(np.float32)/max(n-1,1)
    ip=idx_map.get(date,0); rb=np.full(n,pv-ip,dtype=np.float32)
    cr=np.stack([np.tile(pr[:,None],(1,W)),np.tile(ar[:,None],(1,W)),
                  np.tile(tr[:,None],(1,W)),np.tile(rb[:,None],(1,W))],-1)
    full=np.concatenate([fa,cr],-1)
    m,s=full.mean(axis=0,keepdims=True),full.std(axis=0,keepdims=True)+1e-8
    return (full-m)/s, codes, fa[:,-1,6], fa[:,-1,5], fa[:,-1,7]

def main():
    device = torch.device("cuda")
    print(f"[verify_spatial] device={device}")
    
    m = GRURanker(22,128,1,0.2)
    m.load_state_dict(torch.load(os.path.join(V2_DIR,"checkpoints",CKPT),map_location=device,weights_only=True))
    m.to(device).eval()
    
    df = pd.read_parquet(os.path.join(ROOT,"processed","all_data.parquet"))
    df["trade_date"] = df["trade_date"].astype(str)
    dates_all = sorted(df["trade_date"].unique())
    si = [i for i,d in enumerate(dates_all) if d>="20260201"][0]
    lb = max(0, si-W-1-90); df = df[df["trade_date"]>=dates_all[lb]]
    
    csi = pd.read_csv(os.path.join(ROOT,"data","market","000300.SH.csv"),dtype={"trade_date":str})
    idx_map = dict(zip(csi["trade_date"], csi["pct_chg"]))
    
    stocks = sorted(df["ts_code"].unique())
    np.random.seed(42)
    stocks = sorted(np.random.choice(stocks, N_STOCKS, replace=False))
    series = {ts: add_tech(df[df["ts_code"]==ts].sort_values("trade_date").reset_index(drop=True)) for ts in stocks}
    
    test_dates = [d for d in dates_all if "20260201" <= d <= "20260529"]
    
    # Scan different K and different fusion weights
    Ks = [3, 5, 10, 20]
    alphas = [0.0, 0.1, 0.2, 0.3, 0.5]  # weight on KNN score
    results = {(K, a): [] for K in Ks for a in alphas}
    
    for di, date in enumerate(test_dates):
        if di == len(test_dates)-1: break
        nd = test_dates[di+1]
        f, codes, pv, av, tv = build(date, series, idx_map, W)
        if f is None or len(codes)<30: continue
        
        scores = m(torch.from_numpy(f).to(device)).detach().cpu().numpy()
        
        # Cross-sectional features for KNN
        # Use rank percentiles as features (already in f but easier to compute from raw)
        n = len(codes)
        sx = np.stack([pv, av, tv], axis=1).astype(np.float32)  # [N, 3]
        
        # Z-score standardize cross-sectional features for KNN
        sx = (sx - sx.mean(axis=0)) / (sx.std(axis=0) + 1e-8)
        
        # Filter to stocks with valid next-day returns
        valid_idx = []
        for ci, cc in enumerate(codes):
            r = series[cc][series[cc]["trade_date"]==nd]
            if len(r): valid_idx.append(ci)
        if len(valid_idx) < 30: continue
        
        sl = scores[valid_idx]
        lb_ret = np.array([series[codes[ci]][series[codes[ci]]["trade_date"]==nd]["pct_chg"].values[0] for ci in valid_idx])
        sx_v = sx[valid_idx]  # cross-sectional features for valid stocks only
        
        # Recompute KNN on valid stocks only (needed for consistent indices)
        nn_v = NearestNeighbors(n_neighbors=min(max(Ks)+1, len(valid_idx)), metric="euclidean")
        nn_v.fit(sx_v)
        _, knn_v_idx = nn_v.kneighbors(sx_v)  # [N_valid, maxK+1]
        
        # Compute GRU-only IC
        ic_base = spearmanr(sl, lb_ret).correlation
        
        # For each K and alpha
        for K in Ks:
            if K >= len(valid_idx): continue
            # Exclude self from neighbors (first neighbor is self)
            knn_idx = knn_v_idx[:, 1:K+1]  # [N_valid, K], skip self
            knn_scores = np.mean(sl[knn_idx], axis=1)  # [N_valid]
            
            for alpha in alphas:
                fused = (1-alpha) * sl + alpha * knn_scores
                ic = spearmanr(fused, lb_ret).correlation
                results[(K, alpha)].append(ic)
        
        torch.cuda.empty_cache()
        if (di+1) % 20 == 0:
            print(f"  [{di+1}] base_IC={ic_base:+.4f}", flush=True)
    
    print(f"\n{'='*60}")
    print(f" 验证 A: KNN 空间注意力 proxy")
    base_vals = results[(Ks[0], 0.0)]
    ic_base = np.mean(base_vals) if base_vals else 0
    print(f" Baseline GRU-only IC: {ic_base:+.4f}")
    print(f"\n{'K':>4} {'alpha':>6} {'Mean IC':>10} {'IC gain':>10} {'IC>0':>7}")
    print("-"*45)
    best_gain, best_pair = 0.0, None
    for K in Ks:
        base_K = np.mean(results[(K, 0.0)]) if results[(K, 0.0)] else 0
        for alpha in alphas:
            if alpha == 0: continue
            vals = results[(K, alpha)]
            if not vals: continue
            mean_ic = np.mean(vals)
            gain = mean_ic - base_K
            pos = np.mean([v > 0 for v in vals])
            if gain > best_gain:
                best_gain = gain; best_pair = (K, alpha)
            if abs(gain) > 0.0005:
                print(f"{K:>4} {alpha:>6.1f} {mean_ic:+10.4f} {gain:+10.4f} {pos:>6.1%}")
    print(f"\n 结论: best gain = {best_gain:+.4f} @ K={best_pair[0]}, alpha={best_pair[1]}  "
          f"{'-> 空间注意力有价值' if best_gain > 0.003 else '-> 无显著增益，GRU已捕捉截面信息'}" if best_pair else "")
    print(f"{'='*60}")

if __name__ == "__main__":
    main()
