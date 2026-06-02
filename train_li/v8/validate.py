"""v8/validate.py — Score + IC + Sweep + Backtest for v8 buy model.

1. Precompute scores on 600 sampled stocks (Feb-May 2026)
2. Compute T+1 IC stats
3. Run N/K sweep with Strategy B
"""
import os, sys, gc, glob, numpy as np, pandas as pd, torch
from scipy.stats import spearmanr

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(SCRIPT_DIR)
PARQUET = os.path.join(ROOT, "processed", "all_data.parquet")
INDEX_P = os.path.join(ROOT, "data", "market", "000300.SH.csv")
MONEYFLOW_D = os.path.join(ROOT, "data", "moneyflow")
BASIC_P = os.path.join(ROOT, "data", "basic.csv")

sys.path.insert(0, ROOT)
from v8.models.gru_spatial_v8 import GRURankerSpatialV8

W, START, END = 60, "20260203", "20260529"
N_SAMPLED = 600

RAW = ["open","high","low","close","vol","amount","pct_chg","turnover_rate","volume_ratio","total_mv"]
TECH = ["macd","macd_signal","rsi","bb_width","bb_pct","mom_5","mom_20","vol_20"]
NEW_TIER1 = ["amihud_20","price_pos_20","ret_skew_20"]
MF_COLS = ["mf_flow_hhi","mf_sm_lg_div"]
ALL_TECH = RAW + TECH + NEW_TIER1  # 21 cols before moneyflow
LOG_COLS = ["vol","amount","total_mv"]
PE_CLIP, PB_CLIP = (0.1,500.0), (0.1,50.0)
WINSOR_P = (1,99)
CSI5D_THRESH, RISK_OFF = -1.0, 0.80
N_INDUSTRIES = 111

def add_tech(df):
    c=df["close"].astype(float)
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

def add_new_features(df):
    ret=df["pct_chg"].astype(float)/100.0; amount=df["amount"].astype(float)
    df["amihud_20"]=(np.abs(ret)/np.maximum(amount,1e-8)).rolling(20,min_periods=5).mean()
    c=df["close"].astype(float)
    L20=c.rolling(20,min_periods=5).min(); H20=c.rolling(20,min_periods=5).max()
    df["price_pos_20"]=(c-L20)/np.maximum(H20-L20,1e-8)
    df["ret_skew_20"]=ret.rolling(20,min_periods=10).skew().fillna(0)
    return df

def winsorize_2d(arr): lo=np.percentile(arr,1,axis=0,keepdims=True); hi=np.percentile(arr,99,axis=0,keepdims=True); return np.clip(arr,lo,hi)
def norm_t(arr): m,s=arr.mean(axis=0,keepdims=True),arr.std(axis=0,keepdims=True)+1e-8; return (arr-m)/s
def norm_c(arr): m,s=arr.mean(axis=0,keepdims=True),arr.std(axis=0,keepdims=True)+1e-8; return (arr-m)/s
def rank_pct(arr,N): valid=~np.isnan(arr); out=np.zeros(N,dtype=np.float32); order=np.argsort(np.argsort(arr[valid])) if valid.sum()>=2 else np.array([]); out[valid]=order.astype(np.float32)/max(valid.sum()-1,1); return out

# ========== Load data ==========
print("[validate_v8] Loading data ...")
np.random.seed(42)

# Industry
basic=pd.read_csv(BASIC_P,dtype={"ts_code":str}); basic["industry"]=basic["industry"].fillna("Other")
ind_list=sorted(basic["industry"].unique()); ind2id={ind:i for i,ind in enumerate(ind_list)}
ts2ind=dict(zip(basic["ts_code"],basic["industry"].map(ind2id))); default_ind=ind2id.get("Other",0)

# Moneyflow lookup
print("[validate_v8] Building moneyflow lookup ...")
mf_files=sorted(glob.glob(os.path.join(MONEYFLOW_D,"*.csv")))
mf_lookup={}
for f in mf_files:
    d=os.path.basename(f).replace(".csv","")
    try:
        chunk=pd.read_csv(f,dtype={"ts_code":str,"trade_date":str})
        needed=["ts_code","buy_sm_vol","sell_sm_vol","buy_md_vol","sell_md_vol",
                "buy_lg_vol","sell_lg_vol","buy_elg_vol","sell_elg_vol"]
        avail=[c for c in needed if c in chunk.columns]
        if len(avail)<len(needed)-1: continue  # at least ts_code + 8 vol cols
        chunk=chunk[avail]
        for c in avail:
            if c!="ts_code": chunk[c]=pd.to_numeric(chunk[c],errors="coerce").fillna(0)
        sm_t=chunk["buy_sm_vol"]+chunk["sell_sm_vol"]; md_t=chunk["buy_md_vol"]+chunk["sell_md_vol"]
        lg_t=chunk["buy_lg_vol"]+chunk["sell_lg_vol"]; elg_t=chunk["buy_elg_vol"]+chunk["sell_elg_vol"]
        total=sm_t+md_t+lg_t+elg_t
        ta=np.maximum(total.values,1e-8)
        shares=np.stack([sm_t.values,md_t.values,lg_t.values,elg_t.values],axis=1)/ta[:,None]
        hhi=(shares**2).sum(axis=1)
        sm_net=chunk["buy_sm_vol"]-chunk["sell_sm_vol"]
        lg_net=chunk["buy_lg_vol"]+chunk["buy_elg_vol"]-chunk["sell_lg_vol"]-chunk["sell_elg_vol"]
        div=sm_net-lg_net
        for i,ts in enumerate(chunk["ts_code"]):
            mf_lookup[(ts,str(d))]=(float(hhi[i]),float(div[i]))
    except Exception: continue
print(f"[validate_v8] Moneyflow lookup: {len(mf_lookup)} entries")

# Main data (300-day warmup)
df=pd.read_parquet(PARQUET); df["trade_date"]=df["trade_date"].astype(str)
all_dates=sorted(df["trade_date"].unique())
test_start_idx=all_dates.index(START) if START in all_dates else 0
warmup_cutoff=max(0,test_start_idx-300)
df=df[df["trade_date"]>=all_dates[warmup_cutoff]]
all_stocks=sorted(df["ts_code"].unique())
sampled=sorted(np.random.choice(all_stocks,N_SAMPLED,replace=False))
df=df[df["ts_code"].isin(sampled)]
print(f"[validate_v8] Data: {df['trade_date'].min()}~{df['trade_date'].max()}, {df['ts_code'].nunique()} stocks, {len(df):,} rows")

ret_d,mv_d={},{}
for d in sorted(set(df["trade_date"])):
    sub=df[df["trade_date"]==d]
    ret_d[d]=dict(zip(sub["ts_code"],sub["pct_chg"].astype(float)))
    mv_d[d]=dict(zip(sub["ts_code"],sub["total_mv"].astype(float)))

csi=pd.read_csv(INDEX_P,dtype={"trade_date":str})
csi_map=dict(zip(csi["trade_date"],csi["pct_chg"].astype(float)))
csi_dates=sorted(csi_map.keys())

def get_csi5d(date):
    if date not in csi_dates: return 0.0
    idx=csi_dates.index(date); start=max(0,idx-4)
    return sum(csi_map[csi_dates[i]] for i in range(start,idx+1))

series={}
for ts,sdf in df.groupby("ts_code"):
    sdf=sdf.sort_values("trade_date").reset_index(drop=True).copy().ffill()
    series[ts]=sdf
print(f"[validate_v8] Series loaded: {len(series)} stocks")

# ========== Feature computation ==========
def compute_features(tgt_date, idx_map):
    windows,last_pct,last_amt,last_tvr,last_pe,last_pb,last_cm={},{},{},{},{},{},{}
    ind_ids=[]

    for ts,sdf in series.items():
        sdf_c=sdf[sdf["trade_date"]<=tgt_date]
        if len(sdf_c)<W: continue
        sdf_c=sdf_c.ffill().copy()
        sdf_c=add_tech(sdf_c)
        sdf_c=add_new_features(sdf_c)

        tech_names=ALL_TECH+MF_COLS
        vals_raw=np.zeros((W,len(tech_names)),dtype=np.float32)
        for ci,cn in enumerate(tech_names):
            if cn in sdf_c.columns:
                vals_raw[:,ci]=sdf_c[cn].values.astype(np.float32)[-W:]
        for cn in LOG_COLS:
            if cn in RAW:
                ri=tech_names.index(cn)
                vals_raw[:,ri]=np.log1p(np.maximum(vals_raw[:,ri],0))

        # Moneyflow: per-step values from lookup
        L=len(sdf_c); idx_hhi=tech_names.index("mf_flow_hhi"); idx_div=tech_names.index("mf_sm_lg_div")
        for ti in range(W):
            src_idx=L-W+ti
            if 0<=src_idx<L:
                sd=str(sdf_c["trade_date"].iloc[src_idx])
                mf_key=(ts,sd)
                if mf_key in mf_lookup:
                    vals_raw[ti,idx_hhi]=mf_lookup[mf_key][0]
                    vals_raw[ti,idx_div]=mf_lookup[mf_key][1]

        wc=sdf_c["close"].values.astype(np.float32)[-W:]
        wv=sdf_c["vwap"].values.astype(np.float32)[-W:]
        vg=wc/np.maximum(wv,1e-8)-1
        if np.isnan(vals_raw).any() or np.isnan(vg).any(): continue

        windows[ts]=(vals_raw,vg)
        lr=sdf_c.iloc[-1]
        last_pct[ts]=lr["pct_chg"]; last_amt[ts]=lr["amount"]; last_tvr[ts]=lr["turnover_rate"]
        last_pe[ts]=lr.get("pe",0); last_pb[ts]=lr.get("pb",0); last_cm[ts]=lr.get("circ_mv",0)
        ind_ids.append(ts2ind.get(ts,default_ind))

    code_list=list(windows.keys()); N=len(code_list)
    if N<50: return None,None,None

    idx_pct=idx_map.get(tgt_date,0.0)
    pct_a=np.array([last_pct[c] for c in code_list],dtype=np.float32)
    amt_a=np.array([last_amt[c] for c in code_list],dtype=np.float32)
    tvr_a=np.array([last_tvr[c] for c in code_list],dtype=np.float32)
    pe_a=np.array([last_pe[c] for c in code_list],dtype=np.float32)
    pb_a=np.array([last_pb[c] for c in code_list],dtype=np.float32)
    cm_a=np.array([last_cm[c] for c in code_list],dtype=np.float32)
    cf=np.zeros((N,7),dtype=np.float32)
    cf[:,0]=rank_pct(pct_a,N); cf[:,1]=rank_pct(amt_a,N); cf[:,2]=rank_pct(tvr_a,N)
    cf[:,3]=pct_a-np.float32(idx_pct)
    cf[:,4]=rank_pct(np.clip(pe_a,*PE_CLIP),N); cf[:,5]=rank_pct(np.clip(pb_a,*PB_CLIP),N)
    cf[:,6]=rank_pct(cm_a,N)
    cn=norm_c(cf)

    ta=[]
    for ts_code in code_list:
        vr,vg=windows[ts_code]
        ta.append(np.concatenate([vr,vg[:,None]],axis=1))
    ts_=np.stack(ta,0)
    Ns,Ts,Fs=ts_.shape
    tf=ts_.reshape(-1,Fs); tf=winsorize_2d(tf); ts_=tf.reshape(Ns,Ts,Fs)

    batch,ind_batch=[],[]
    for i in range(N):
        t=norm_t(ts_[i])
        ct=np.tile(cn[i],(W,1))
        batch.append(np.concatenate([t,ct],-1))
    ind_arr=np.array([ts2ind.get(c,default_ind) for c in code_list],dtype=np.int64)

    return np.stack(batch),code_list,ind_arr


# ========== Load model ==========
device=torch.device("cuda")
ckpt_listmle=os.path.join(SCRIPT_DIR,"checkpoints","v8_listmle_d32_K5_H128_L1_D0.2_lr0.0003_N1024_ind111.pt")
print(f"[validate_v8] Loading v8 listmle checkpoint ...")
m=GRURankerSpatialV8(31,128,1,0.2,d_proj=32,K=5,n_industries=N_INDUSTRIES).to(device).eval()
state=torch.load(ckpt_listmle,map_location=device,weights_only=True)
state={k.replace("_orig_mod.",""):v for k,v in state.items()}
m.load_state_dict(state)

# ========== Score ==========
print(f"\n[validate_v8] Scoring test dates ...")
test_dates=[d for d in all_dates if START<=d<=END]
records=[]

for di,d in enumerate(test_dates):
    feat,codes,ind_arr=compute_features(d,csi_map)
    if feat is None:
        if di<2: print(f"  [DEBUG] {d}: no features")
        continue
    if di<2: print(f"  [DEBUG] {d}: {feat.shape} for {len(codes)} stocks")
    t=torch.from_numpy(feat).float().to(device)
    ind_t=torch.from_numpy(ind_arr).long().to(device)
    with torch.no_grad():
        scores=m(t,ind_t).cpu().numpy()
    for i,c in enumerate(codes):
        records.append({"trade_date":d,"ts_code":c,"score":float(scores[i])})
    if (di+1)%20==0: print(f"  [{di+1}/{len(test_dates)}] {d}")

scores_df=pd.DataFrame(records)
scores_df.to_parquet(os.path.join(SCRIPT_DIR,"results","daily_scores_v8.parquet"),index=False)
print(f"[validate_v8] Scores saved: {len(records)} records")

# ========== IC ==========
sd={}
for r in records: sd.setdefault(r["trade_date"],{})[r["ts_code"]]=r["score"]
ics=[]
for i in range(len(all_dates)-1):
    fd,rd=all_dates[i],all_dates[i+1]
    if fd not in sd or rd not in ret_d: continue
    ss=sd[fd]; rr=ret_d[rd]
    common=[ts for ts in ss if ts in rr]
    if len(common)<50: continue
    s_val=[ss[ts] for ts in common]; r_val=[rr[ts] for ts in common]
    ic=spearmanr(s_val,r_val).correlation
    if not np.isnan(ic): ics.append(ic)

print(f"\nv8 Buy (listmle): IC={np.mean(ics):+.4f} std={np.std(ics):.4f} >0={np.mean([x>0 for x in ics]):.1%} n={len(ics)}")

# ========== Sweep ==========
print(f"\n{'='*60}")
print(" Sweep: v8 Buy (listmle) — N_hold x sell_k with Strategy B")
print(f"{'='*60}")

test_list=sorted(sd.keys())
sweep_results=[]
for N in [5,6,7,8,9,10,15,20]:
    for K in [2,3,5,8,10]:
        if K>=N: continue
        holdings=None; daily_p,daily_c=[],[]
        for si in range(len(test_list)-1):
            fd=test_list[si]; rd=test_list[si+1]
            if fd not in sd or rd not in ret_d: continue
            ss=sd[fd]; rr=ret_d[rd]
            sorted_s=sorted(ss.items(),key=lambda x:x[1],reverse=True)
            top=[c for c,_ in sorted_s if c in rr]
            if len(top)<N: continue
            csi5d=get_csi5d(fd)
            pos=RISK_OFF if csi5d<CSI5D_THRESH else 1.0
            target=max(1,int(round(N*pos)))
            if holdings is None:
                holdings=top[:target]
            else:
                held_s=[(c,ss.get(c,-1e6)) for c in holdings if c in ss]
                held_s.sort(key=lambda x:x[1],reverse=True)
                to_sell={c for c,_ in held_s[-K:]} if len(held_s)>K else set()
                extra=len(holdings)-target
                if extra>0:
                    cand=[c for c,_ in held_s if c not in to_sell]
                    for c in cand[-extra:]: to_sell.add(c)
                holdings=[c for c in holdings if c not in to_sell]
                held_set=set(holdings)
                for c in top:
                    if len(holdings)>=target: break
                    if c not in held_set: holdings.append(c)
            pr=np.mean([rr.get(c,0.0) for c in holdings]) if holdings else 0.0
            cr=csi_map.get(rd,0.0)
            daily_p.append(pr); daily_c.append(cr)
        if daily_p:
            cum=np.sum(daily_p); cum_c=np.sum(daily_c)
            sweep_results.append({"N":N,"K":K,"cum":round(cum,4),"excess":round(cum-cum_c,4),"days":len(daily_p)})

sweep_df=pd.DataFrame(sweep_results).sort_values("cum",ascending=False)
sweep_df.to_csv(os.path.join(SCRIPT_DIR,"results","sweep_v8.csv"),index=False)

print(f"{'N':>4} {'K':>4} {'Cum':>8} {'Excess':>8} {'Days':>5}")
for _,r in sweep_df.head(12).iterrows():
    print(f"{r['N']:>4.0f} {r['K']:>4.0f} {r['cum']:>+7.2f}% {r['excess']:>+7.2f}% {r['days']:>5}")

best=sweep_df.iloc[0]
print(f"\nBest: N={int(best['N'])}, K={int(best['K'])}, cum={best['cum']:+.2f}%, excess={best['excess']:+.2f}%")
