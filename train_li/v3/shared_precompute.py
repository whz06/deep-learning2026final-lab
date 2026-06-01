"""Precompute daily model scores + CSI300 signals for strategy testing.
Saves to v3/results/benchmark_data.parquet so strategy tests don't recompute.
"""
import os, sys, numpy as np, pandas as pd, torch

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(SCRIPT_DIR)
V2_DIR = os.path.join(ROOT, "v2")
PARQUET_PATH = os.path.join(ROOT, "processed", "all_data.parquet")
INDEX_PATH   = os.path.join(ROOT, "data", "market", "000300.SH.csv")
CKPT_DIR = os.path.join(V2_DIR, "checkpoints")
OUT_PATH = os.path.join(SCRIPT_DIR, "results", "benchmark_data.parquet")

sys.path.insert(0, V2_DIR)
from models.gru import GRURanker

W = 60; CKPT = "gru_gru_hidden_size=128_num_layers=1_dropout=0.2_lr=0.0003.pt"
RAW = ["open","high","low","close","vol","amount","pct_chg","turnover_rate","volume_ratio","total_mv"]
TECH = ["macd","macd_signal","rsi","bb_width","bb_pct","mom_5","mom_20","vol_20"]

def add_tech(df):
    c=df["close"].astype(float)
    e12,e26=c.ewm(span=12,adjust=False).mean(),c.ewm(span=26,adjust=False).mean()
    df["macd"]=e12-e26; df["macd_signal"]=df["macd"].ewm(span=9,adjust=False).mean()
    d=c.diff(); g=d.clip(lower=0); l=(-d).clip(lower=0)
    rs=g.ewm(alpha=1/14,adjust=False).mean()/(l.ewm(alpha=1/14,adjust=False).mean()+1e-8)
    df["rsi"]=100-100/(1+rs)
    m20,s20=c.rolling(20).mean(),c.rolling(20).std()
    df["bb_width"]=2*s20/(m20+1e-8); df["bb_pct"]=(c-(m20-2*s20))/(4*s20+1e-8)
    df["mom_5"]=c/c.shift(5)-1; df["mom_20"]=c/c.shift(20)-1
    df["vol_20"]=c.pct_change().rolling(20).std()
    return df

def build(date, series, idx_map):
    feats,codes,vols=[],[],[]
    for ts,sdf in series.items():
        sd=sdf[sdf["trade_date"]<=date]
        if len(sd)<W+1: continue
        sw=sd.iloc[-W-1:]
        rv=sw[RAW+TECH].values.astype(np.float32)
        if np.isnan(rv).any(): continue
        feats.append(rv[-W:]); codes.append(ts); vols.append(sw["vol_20"].values[-1])
    if not feats: return None,None,None,None
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
    return (full-m)/s, codes, pv, np.array(vols)

def main():
    device=torch.device("cuda")
    m=GRURanker(22,128,1,0.2)
    m.load_state_dict(torch.load(os.path.join(CKPT_DIR,CKPT),map_location=device,weights_only=True))
    m.to(device).eval()

    df=pd.read_parquet(PARQUET_PATH); df["trade_date"]=df["trade_date"].astype(str)
    dates_all=sorted(df["trade_date"].unique())
    tc=[d for d in dates_all if d>="20260201"]
    lb=max(0,dates_all.index(tc[0])-W-1-20)
    df=df[df["trade_date"]>=dates_all[lb]]

    csi=pd.read_csv(INDEX_PATH,dtype={"trade_date":str})
    idx_map=dict(zip(csi["trade_date"],csi["pct_chg"]))
    csi_dates=csi["trade_date"].tolist()
    csi_vals=csi["pct_chg"].tolist()

    stocks=sorted(df["ts_code"].unique())
    np.random.seed(42); stocks=sorted(np.random.choice(stocks,600,replace=False))
    print(f"[precompute] {len(stocks)} stocks")

    series={ts:add_tech(df[df["ts_code"]==ts].sort_values("trade_date").reset_index(drop=True)) for ts in stocks}
    test_dates=[d for d in dates_all if "20260201"<=d<="20260531"]

    records=[]
    for di,date in enumerate(test_dates):
        if di==len(test_dates)-1: break
        nd=test_dates[di+1]
        f,c,pv,vol=build(date,series,idx_map)
        if f is None: continue
        scores=m(torch.from_numpy(f).to(device)).detach().cpu().numpy()
        vl,vi=[],[]
        for ci,cc in enumerate(c):
            r=series[cc][series[cc]["trade_date"]==nd]
            if len(r)>0: vl.append(r["pct_chg"].values[0]); vi.append(ci)
        if not vi: continue
        sl=scores[vi]; lb=vl

        di_idx=csi_dates.index(nd) if nd in csi_dates else -1
        idx_ret=idx_map.get(nd,0)

        # CSI300 statistics (backward-looking only)
        csi5d_sum=0; csi20d_sum=0; csi10d_sum=0
        csi20vol=0; csi10vol=0; csi5vol=0; csi60vol=0
        if di_idx>=0:
            if di_idx>=4:  csi5d_sum=float(np.sum(csi_vals[di_idx-4:di_idx+1]))
            if di_idx>=19: csi20d_sum=float(np.sum(csi_vals[di_idx-19:di_idx+1]))
            if di_idx>=9:  csi10d_sum=float(np.sum(csi_vals[di_idx-9:di_idx+1]))
            if di_idx>=4:  csi5vol=float(np.std(csi_vals[max(0,di_idx-4):di_idx+1]))
            if di_idx>=9:  csi10vol=float(np.std(csi_vals[max(0,di_idx-9):di_idx+1]))
            if di_idx>=19: csi20vol=float(np.std(csi_vals[max(0,di_idx-19):di_idx+1]))
            if di_idx>=59: csi60vol=float(np.std(csi_vals[max(0,di_idx-59):di_idx+1]))

        top20=int(len(sl)*0.2)
        top_idx=np.argsort(sl)[-top20:]

        # Score distribution statistics (for dispersion defense)
        score_std=float(sl.std())
        score_mean=float(sl.mean())
        z=(sl-score_mean)/(score_std+1e-8)
        score_skew=float((z**3).mean())
        score_min=float(sl.min()); score_max=float(sl.max())
        top20_scores=sl[top_idx]
        top20_score_mean=float(top20_scores.mean())
        top20_score_std=float(top20_scores.std())
        top20_score_range=float(top20_scores.max()-top20_scores.min())
        z_scores=(sl-score_mean)/(score_std+1e-8)
        n_above_z2=int((z_scores>2).sum())

        records.append({
            "date": nd,
            "idx_ret": float(idx_ret),
            "top20_mean": float(np.mean([lb[i] for i in top_idx])),
            "top20_std": float(np.std([lb[i] for i in top_idx])),
            "top20_median": float(np.median([lb[i] for i in top_idx])),
            "top20_min": float(np.min([lb[i] for i in top_idx])),
            "top20_max": float(np.max([lb[i] for i in top_idx])),
            "all_mean": float(np.mean(lb)),
            "all_std": float(np.std(lb)),
            "n_stocks": int(len(lb)),
            "ic": float(np.corrcoef(sl,lb)[0,1]),
            "csi5d": float(csi5d_sum),
            "csi10d": float(csi10d_sum),
            "csi20d": float(csi20d_sum),
            "csi5vol": float(csi5vol),
            "csi10vol": float(csi10vol),
            "csi20vol": float(csi20vol),
            "csi60vol": float(csi60vol),
            "score_mean": score_mean,
            "score_std": score_std,
            "score_skew": score_skew,
            "score_min": score_min,
            "score_max": score_max,
            "top20_score_mean": top20_score_mean,
            "top20_score_std": top20_score_std,
            "top20_score_range": top20_score_range,
            "n_above_z2": int(n_above_z2),
        })
        torch.cuda.empty_cache()

    rec=pd.DataFrame(records)
    rec.to_parquet(OUT_PATH, index=False)
    print(f"[precompute] Saved {len(rec)} days to {OUT_PATH}")
    print(f"  Model cumulative: {np.prod(1+rec['top20_mean'].values/100):.4f}")
    print(f"  CSI300 cumulative: {np.prod(1+rec['idx_ret'].values/100):.4f}")

if __name__=="__main__":
    main()
