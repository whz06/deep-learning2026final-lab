"""Why C/A+C underperform: data-driven proof."""
import os, sys, numpy as np, pandas as pd, torch

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(SCRIPT_DIR)
V2_DIR = os.path.join(ROOT, "v2")
PARQUET_PATH = os.path.join(ROOT, "processed", "all_data.parquet")
INDEX_PATH   = os.path.join(ROOT, "data", "market", "000300.SH.csv")
CKPT_DIR = os.path.join(V2_DIR, "checkpoints")

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
    feats, codes, vol20s=[],[],[]
    for ts, sdf in series.items():
        sd=sdf[sdf["trade_date"]<=date]
        if len(sd)<W+1: continue
        sw=sd.iloc[-W-1:]
        rv=sw[RAW+TECH].values.astype(np.float32)
        if np.isnan(rv).any(): continue
        feats.append(rv[-W:]); codes.append(ts); vol20s.append(sw["vol_20"].values[-1])
    if not feats: return None,None,None,None
    fa=np.stack(feats,0); n=len(feats)
    pv=fa[:,-1,6]; av=fa[:,-1,5]; tv=fa[:,-1,7]
    pr=np.argsort(np.argsort(pv)).astype(np.float32)/max(n-1,1)
    ar=np.argsort(np.argsort(av)).astype(np.float32)/max(n-1,1)
    tr=np.argsort(np.argsort(tv)).astype(np.float32)/max(n-1,1)
    ip=idx_map.get(date,0); rb=np.full(n,pv-ip,dtype=np.float32)
    cr=np.stack([np.tile(pr[:,None],(1,W)),np.tile(ar[:,None],(1,W)),
                  np.tile(tr[:,None],(1,W)),np.tile(rb[:,None],(1,W))],-1)
    full=np.concatenate([fa,cr],-1)
    m,s=full.mean(axis=0,keepdims=True),full.std(axis=0,keepdims=True)+1e-8
    return (full-m)/s, codes, pv, np.array(vol20s)

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

    idx_map=dict(zip(pd.read_csv(INDEX_PATH,dtype={"trade_date":str})["trade_date"],
                      pd.read_csv(INDEX_PATH,dtype={"trade_date":str})["pct_chg"]))

    stocks = sorted(df["ts_code"].unique())
    np.random.seed(42)
    stocks = sorted(np.random.choice(stocks, 500, replace=False))
    print(f"[speed] Sampled {len(stocks)} stocks")
    series = {ts: add_tech(df[df["ts_code"] == ts].sort_values("trade_date").reset_index(drop=True)) for ts in stocks}

    test_dates=[d for d in dates_all if "20260201"<=d<="20260531"]

    print("Computing...")
    dd={}
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
        dd[date]={"scores":scores[vi],"codes":[c[i] for i in vi],"labels":np.array(vl),"vol_20":vol[vi]}
        torch.cuda.empty_cache()

    # CSI300 5d
    csi_d=sorted(dd.keys())
    csi5d={}
    for i,d in enumerate(csi_d):
        if i>=5: csi5d[d]=float(np.prod(1+np.array([idx_map.get(csi_d[j],0) for j in range(i-4,i+1)])/100)-1)*100
        else: csi5d[d]=0

    print(f"\n{'Date':<12} {'State':>8} {'All%':>7} {'AlphaHi%':>7} {'LoVol%':>7} {'Alpha-LoVol':>12} {'IC':>7}")
    print("-"*78)

    bear_all, bear_alpha, bear_lovol = [],[],[]
    bull_all, bull_alpha, bull_lovol = [],[],[]

    for date in csi_d:
        d=dd[date]; lv=d["labels"]; sc=d["scores"]; v20=d["vol_20"]
        c5=csi5d[date]
        st="BEAR" if c5<-2 else ("BULL" if c5>1 else "NEUTRAL")
        am=lv.mean()
        ah=lv[np.argsort(sc)[-int(len(sc)*0.2):]].mean()
        lv20=lv[v20<=np.percentile(v20,20)].mean()
        ic=np.corrcoef(sc,lv)[0,1]
        diff=ah-lv20
        print(f"{date:<12} {st:>8} {am:>+6.2f}% {ah:>+6.2f}% {lv20:>+6.2f}% {diff:>+11.2f}% {ic:>+7.4f}")

        if st=="BEAR":
            bear_all.append(am); bear_alpha.append(ah); bear_lovol.append(lv20)
        elif st=="BULL":
            bull_all.append(am); bull_alpha.append(ah); bull_lovol.append(lv20)

    print(f"\n{'='*78}")
    print(f"AVERAGES BY MARKET STATE")
    print(f"{'':<12} {'All%':>7} {'AlphaHi%':>7} {'LoVol%':>7}")
    if bear_all:
        print(f"BEAR ({len(bear_all)}d)   {np.mean(bear_all):>+6.2f}% {np.mean(bear_alpha):>+6.2f}% {np.mean(bear_lovol):>+6.2f}%")
        print(f"  → C策略: 放弃Alpha选LoVol → 多亏 {np.mean(bear_alpha)-np.mean(bear_lovol):.2f}%")
    if bull_all:
        print(f"BULL ({len(bull_all)}d)   {np.mean(bull_all):>+6.2f}% {np.mean(bull_alpha):>+6.2f}% {np.mean(bull_lovol):>+6.2f}%")
        print(f"  → C策略: 正常选Alpha     → 多赚 {np.mean(bull_alpha)-np.mean(bull_lovol):+.2f}%")

    # The critical test: did low-vol stocks actually hold up better in bear markets?
    print(f"\n  KEY QUESTION: In BEAR markets, are LoVol stocks safer?")
    if bear_all:
        lv_delta = np.mean(bear_lovol) - np.mean(bear_all)
        alpha_delta = np.mean(bear_alpha) - np.mean(bear_all)
        print(f"  LoVol vs All:  {lv_delta:+.2f}%  (positive = safer)")
        print(f"  Alpha vs All:  {alpha_delta:+.2f}%")
        if lv_delta <= 0 and alpha_delta > lv_delta:
            print(f"  → LoVol does NOT protect. Alpha still outperforms LoVol even in bear markets.")
            print(f"  → Switching from Alpha to LoVol in bear markets DESTROYS returns.")

if __name__=="__main__":
    main()
