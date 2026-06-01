"""Strategy analysis: compare model vs CSI300, test filters & sizing."""
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
    feats, codes, vol20s, betas = [], [], [], []
    for ts, sdf in series.items():
        sd = sdf[sdf["trade_date"]<=date]
        if len(sd)<W+1: continue
        sw = sd.iloc[-W-1:]
        rv = sw[RAW+TECH].values.astype(np.float32)
        if np.isnan(rv).any(): continue
        feats.append(rv[-W:]); codes.append(ts); vol20s.append(sw["vol_20"].values[-1])
        betas.append(np.nan if len(sd)<50 else np.corrcoef(sd["pct_chg"].iloc[-50:],
            sd["close"].pct_change().iloc[-50:])[0,1] if len(sd)>=50 and sd["pct_chg"].iloc[-50:].std()>0 else np.nan)
    if not feats: return None,None,None,None,None
    fa = np.stack(feats,0); n=len(feats)
    pv,av,tv = fa[:,-1,6], fa[:,-1,5], fa[:,-1,7]
    pr = np.argsort(np.argsort(pv)).astype(np.float32)/max(n-1,1)
    ar = np.argsort(np.argsort(av)).astype(np.float32)/max(n-1,1)
    tr = np.argsort(np.argsort(tv)).astype(np.float32)/max(n-1,1)
    ip = idx_map.get(date,0); rb = np.full(n,pv-ip,dtype=np.float32)
    cr = np.stack([np.tile(pr[:,None],(1,W)),np.tile(ar[:,None],(1,W)),
                    np.tile(tr[:,None],(1,W)),np.tile(rb[:,None],(1,W))],-1)
    full = np.concatenate([fa,cr],-1)
    m,s = full.mean(axis=0,keepdims=True), full.std(axis=0,keepdims=True)+1e-8
    return (full-m)/s, codes, pv, np.array(vol20s), np.array(betas)

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

    idx=pd.read_csv(INDEX_PATH,dtype={"trade_date":str})
    idx_map=dict(zip(idx["trade_date"], idx["pct_chg"]))

    stocks=sorted(df["ts_code"].unique())
    np.random.seed(42)
    stocks=sorted(np.random.choice(stocks,400,replace=False))
    print(f"[speed] {len(stocks)} stocks")

    series={ts:add_tech(df[df["ts_code"]==ts].sort_values("trade_date").reset_index(drop=True)) for ts in stocks}
    test_dates=[d for d in dates_all if "20260201"<=d<="20260531"]

    records = []
    for di,date in enumerate(test_dates):
        if di==len(test_dates)-1: break
        nd=test_dates[di+1]
        f,c,pv,vol,betas=build(date,series,idx_map)
        if f is None: continue
        scores=m(torch.from_numpy(f).to(device)).detach().cpu().numpy()
        vl,vi=[],[]
        for ci,cc in enumerate(c):
            r=series[cc][series[cc]["trade_date"]==nd]
            if len(r)>0: vl.append(r["pct_chg"].values[0]); vi.append(ci)
        if not vi: continue
        sl=scores[vi]; lb=vl; v20=vol[vi]; bt=betas[vi]
        idx_ret=idx_map.get(nd,0)

        # CSI300 past-N stats
        di_idx=idx["trade_date"].tolist().index(nd) if nd in idx["trade_date"].tolist() else -1
        csi5d_abs=max(0,di_idx-4) if di_idx>=0 else di_idx
        csi20vol = np.std(list(idx_map.values())[max(0,di_idx-19):di_idx+1]) if di_idx>=0 else 0
        csi5d = np.sum(list(idx_map.values())[max(0,di_idx-4):di_idx+1]) if di_idx>=0 else 0

        top20=int(len(sl)*0.2)
        top_idx=np.argsort(sl)[-top20:]
        records.append({
            "date":nd, "idx_ret":idx_ret,
            "top20_mean":np.mean([lb[i] for i in top_idx]),
            "top20_median":np.median([lb[i] for i in top_idx]),
            "top20_worst":np.min([lb[i] for i in top_idx]),
            "all_mean":np.mean(lb),
            "ic":np.corrcoef(sl,lb)[0,1],
            "csi5d":csi5d, "csi20vol":csi20vol,
            "top_beta":np.nanmean(bt[top_idx]),
        })
        torch.cuda.empty_cache()

    rec=pd.DataFrame(records)
    rec["excess"]=rec["top20_mean"]-rec["idx_ret"]

    # ==== 1. How many days does model beat CSI300? How bad are the worst losses? ====
    print("="*70)
    print("SECTION 1: MODEL vs CSI300 — RAW PERFORMANCE")
    print("="*70)
    win=(rec["excess"]>0).sum()
    print(f"  Beat CSI300: {win}/{len(rec)} days ({win/len(rec)*100:.0f}%)")
    print(f"  Mean excess:  {rec['excess'].mean():+.2f}%")
    print(f"  Mean top20:   {rec['top20_mean'].mean():+.2f}%  (CSI300: {rec['idx_ret'].mean():+.2f}%)")

    print(f"\n  WORST 10 MODEL DAYS:")
    worst=rec.nsmallest(10,"top20_mean")
    for _,r in worst.iterrows():
        print(f"    {r['date']}  top20={r['top20_mean']:+.2f}%  CSI300={r['idx_ret']:+.2f}%  csi5d={r['csi5d']:+.1f}%  vol={r['csi20vol']:.3f}")

    # compound return
    cum_model = np.prod(1+rec["top20_mean"].values/100)
    cum_index = np.prod(1+rec["idx_ret"].values/100)
    print(f"\n  Cumulative: Model={cum_model:.4f} ({cum_model-1:+.1%})  CSI300={cum_index:.4f} ({cum_index-1:+.1%})")

    # ==== 2. Decompose by market condition ====
    print(f"\n{'='*70}")
    print("SECTION 2: DECOMPOSE BY MARKET CONDITION")
    print("="*70)

    for label, mask in [
        ("ALL", slice(None)),
        ("CSI300 next DOWN <-2%", rec["idx_ret"]<-2),
        ("CSI300 next DOWN -1~-2%", (rec["idx_ret"]>=-2)&(rec["idx_ret"]<-1)),
        ("CSI300 next FLAT -1~1%", (rec["idx_ret"]>=-1)&(rec["idx_ret"]<1)),
        ("CSI300 next UP +1~+2%", (rec["idx_ret"]>=1)&(rec["idx_ret"]<2)),
        ("CSI300 next UP >+2%", rec["idx_ret"]>=2),
        ("CSI300 prev5d DOWN <-3%", rec["csi5d"]<-3),
        ("CSI300 prev5d UP >+3%", rec["csi5d"]>3),
        ("CSI300 HI-VOL (top30%)", rec["csi20vol"]>=rec["csi20vol"].quantile(0.7)),
    ]:
        sub=rec[mask]
        if len(sub)==0: continue
        print(f"  {label:<30} n={len(sub):>3}  top20={sub['top20_mean'].mean():+5.2f}%  CSI300={sub['idx_ret'].mean():+5.2f}%  excess={sub['excess'].mean():+5.2f}%")

    # ==== 3. Can we predict BAD days? ====
    print(f"\n{'='*70}")
    print("SECTION 3: CAN WE PREDICT TERRIBLE MODEL DAYS?")
    print("="*70)

    # Define "disaster" as model losing >2%
    disaster = rec["top20_mean"] <-2
    nd = len(disaster)
    dp = disaster.sum()
    print(f"  Disaster days (top20<-2%): {dp}/{nd} ({dp/nd*100:.0f}%)")

    # Using csi5d < -2% as filter
    bear_filter = rec["csi5d"] < -2
    if bear_filter.sum()>0:
        dd = rec[bear_filter]["top20_mean"]
        print(f"  If csi5d<-2% (n={bear_filter.sum()}): top20 mean={dd.mean():+.2f}% (50% of disaster days from this group)")

    # Using csi20vol > threshold
    for q in [0.6,0.7,0.8,0.9]:
        vol_filt=rec["csi20vol"]>=rec["csi20vol"].quantile(q)
        if vol_filt.sum()>0:
            dd=rec[vol_filt]["top20_mean"]
            di=disaster[vol_filt].sum()
            print(f"  csi20vol > P{q*100:.0f} (n={vol_filt.sum()}): top20={dd.mean():+.2f}%, disaster={di}")

    # Correlation between features and next-day model return
    print(f"\n  Correlations with model top20 next-day return:")
    for col in ["csi5d", "csi20vol"]:
        cc=rec[col].corr(rec["top20_mean"])
        print(f"    {col:>12}: r={cc:+.4f} (predictive?)")

    # ==== 4. Strategy simulations ====
    print(f"\n{'='*70}")
    print("SECTION 4: STRATEGY SIMULATIONS (80% min position)")
    print("="*70)

    def sim(name, rec, weight_series):
        """weight_series: 0=all cash (but min 20%?), 1=full model"""
        # For competition: must be >=80% position. So cash_stake in [0, 0.2]
        # weight: 1.0 -> 100% position, 0.8 -> 80% position
        pos_weight = 0.8 + 0.2 * np.clip(weight_series, 0, 1)
        daily = rec["top20_mean"].values * pos_weight + rec["idx_ret"].values * (1-pos_weight)
        cum = np.prod(1+daily/100)
        sharpe = daily.mean()/daily.std()*np.sqrt(250) if daily.std()>0 else 0
        maxdd = 0; peak=1
        for d in daily:
            peak=max(peak, peak*(1+d/100)); maxdd=min(maxdd, (peak*(1+d/100))/peak-1)
        return cum-1, daily.mean(), daily.std(), sharpe, maxdd

    # Baseline: always 100% (n=20)
    weight=np.ones(len(rec))
    b_ret,b_mu,b_std,b_sh,b_mdd=sim("baseline",rec,weight)

    # Strategy 1: reduce to 80% on high-vol days
    def vol_filter(vol,q):
        w=np.ones(len(vol))
        w[vol>=np.quantile(vol,q)]=0.0
        return w
    for q in [0.6,0.7,0.8,0.9]:
        r,m,sd,sh,mdd=sim(f"HI-VOL risk-off P{q*100:.0f}",rec,vol_filter(rec["csi20vol"].values,q))
        print(f"  HI-VOL off(P{q*100:.0f}): ret={r:+.1%} sharpe={sh:+.2f} maxdd={mdd:+.1%}  vs baseline: {r-b_ret:+.1%}")

    # Strategy 2: reduce to 80% on negative momentum days
    for thresh in [-5,-3,-2,-1]:
        def mom_filter(csi5d,t):
            w=np.ones(len(csi5d))
            w[csi5d<t]=0.0
            return w
        r,m,sd,sh,mdd=sim(f"CSI5d<thresh {thresh}",rec,mom_filter(rec["csi5d"].values,thresh))
        print(f"  CSI5d<{thresh:+d}% off: ret={r:+.1%} sharpe={sh:+.2f} maxdd={mdd:+.1%}  vs baseline: {r-b_ret:+.1%}")

    # Strategy 3: use CSI300 beta to scale
    def beta_filter(rec):
        w=np.ones(len(rec))
        beta=rec["top_beta"].values
        beta=np.nan_to_num(beta,nan=1.0)
        w=1.0/np.clip(np.abs(beta),0.5,3.0)
        return w
    r,m,sd,sh,mdd=sim("beta-scaled",rec,beta_filter(rec))
    print(f"  Beta-scaled:        ret={r:+.1%} sharpe={sh:+.2f} maxdd={mdd:+.1%}  vs baseline: {r-b_ret:+.1%}")

    print(f"\n  BASELINE:    ret={b_ret:+.1%} sharpe={b_sh:+.2f} maxdd={b_mdd:+.1%}")

    # ==== 5. The kill shot: worst single-day losses ====
    print(f"\n{'='*70}")
    print("SECTION 5: WORST SINGLE DAY ANALYSIS")
    for _,r in rec.nsmallest(5,"excess").iterrows():
        print(f"  {r['date']}  top20={r['top20_mean']:+.2f}%  CSI300={r['idx_ret']:+.2f}%  "
              f"excess={r['excess']:+.2f}%  csi5d={r['csi5d']:+.1f}%  vol20={r['csi20vol']:.4f}")

if __name__=="__main__":
    main()
