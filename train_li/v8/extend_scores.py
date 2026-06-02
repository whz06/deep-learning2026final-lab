"""v8/extend_scores.py — 扩展 v7 Spatial 打分到 1 月 5 日起。

读取 v7 Spatial 模型，对 Jan 2026 (1/5 - 1/30) 生成打分，
追加到 v7/results/daily_scores_spatial_t1.parquet。
"""
import os, sys, glob, numpy as np, pandas as pd, torch

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(SCRIPT_DIR)
PARQUET = os.path.join(ROOT, "processed", "all_data.parquet")
INDEX_P = os.path.join(ROOT, "data", "market", "000300.SH.csv")
MONEYFLOW_D = os.path.join(ROOT, "data", "moneyflow")
BASIC_P = os.path.join(ROOT, "data", "basic.csv")
CKPT = os.path.join(ROOT, "v7", "checkpoints", "gru_spatial_v7_d32_K5_H128_L1_D0.2_lr0.0003_N1024.pt")
OUT = os.path.join(ROOT, "v7", "results", "daily_scores_spatial_t1.parquet")

sys.path.insert(0, ROOT)
from v7.models.gru_spatial import GRURankerSpatial

W, START, END = 60, "20260105", "20260130"
DIMS = 26
RAW = ["open","high","low","close","vol","amount","pct_chg","turnover_rate","volume_ratio","total_mv"]
TECH = ["macd","macd_signal","rsi","bb_width","bb_pct","mom_5","mom_20","vol_20"]
LOG_COLS = ["vol","amount","total_mv"]
PE_CLIP, PB_CLIP = (0.1,500.0), (0.1,50.0)

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

def winsorize_2d(arr): lo=np.percentile(arr,1,axis=0,keepdims=True); hi=np.percentile(arr,99,axis=0,keepdims=True); return np.clip(arr,lo,hi)
def norm_t(arr): m,s=arr.mean(axis=0,keepdims=True),arr.std(axis=0,keepdims=True)+1e-8; return (arr-m)/s
def norm_c(arr): m,s=arr.mean(axis=0,keepdims=True),arr.std(axis=0,keepdims=True)+1e-8; return (arr-m)/s
def rank_pct(arr,N): valid=~np.isnan(arr); out=np.zeros(N,dtype=np.float32); order=np.argsort(np.argsort(arr[valid])) if valid.sum()>=2 else np.array([]); out[valid]=order.astype(np.float32)/max(valid.sum()-1,1); return out

# Load data (300-day warmup)
print("[extend] Loading data ...")
df=pd.read_parquet(PARQUET); df["trade_date"]=df["trade_date"].astype(str)
all_dates=sorted(df["trade_date"].unique())
start_idx=all_dates.index(START) if START in all_dates else 0
warmup=max(0,start_idx-300)
df=df[df["trade_date"]>=all_dates[warmup]]
stocks=sorted(df["ts_code"].unique())

csi=pd.read_csv(INDEX_P,dtype={"trade_date":str})
csi_map=dict(zip(csi["trade_date"],csi["pct_chg"].astype(float)))

series={}
for ts,sdf in df.groupby("ts_code"):
    sdf=sdf.sort_values("trade_date").reset_index(drop=True).copy().ffill()
    series[ts]=sdf
print(f"[extend] {len(series)} stocks loaded")

# Model
device=torch.device("cuda")
m=GRURankerSpatial(DIMS,128,1,0.2,d_proj=32,K=5).to(device).eval()
state=torch.load(CKPT,map_location=device,weights_only=True)
state={k.replace("_orig_mod.",""):v for k,v in state.items()}
m.load_state_dict(state)
print("[extend] v7 Spatial loaded")

# Compute scores
test_dates=[d for d in all_dates if START<=d<=END]
print(f"[extend] Scoring {len(test_dates)} dates ...")
records=[]

for di,d in enumerate(test_dates):
    windows,last_pct,last_amt,last_tvr,last_pe,last_pb,last_cm={},{},{},{},{},{},{}
    for ts,sdf in series.items():
        sdf_c=sdf[sdf["trade_date"]<=d]
        if len(sdf_c)<W: continue
        sdf_c=sdf_c.ffill().copy()
        sdf_c=add_tech(sdf_c)
        vals_raw=sdf_c[RAW+TECH].values.astype(np.float32)[-W:]
        for cn in LOG_COLS:
            if cn in RAW: vals_raw[:,RAW.index(cn)]=np.log1p(np.maximum(vals_raw[:,RAW.index(cn)],0))
        wc=sdf_c["close"].values.astype(np.float32)[-W:]
        wv=sdf_c["vwap"].values.astype(np.float32)[-W:]
        vg=wc/np.maximum(wv,1e-8)-1
        if np.isnan(vals_raw).any() or np.isnan(vg).any(): continue
        windows[ts]=(vals_raw,vg)
        lr=sdf_c.iloc[-1]
        last_pct[ts]=lr["pct_chg"]; last_amt[ts]=lr["amount"]; last_tvr[ts]=lr["turnover_rate"]
        last_pe[ts]=lr.get("pe",0); last_pb[ts]=lr.get("pb",0); last_cm[ts]=lr.get("circ_mv",0)

    code_list=list(windows.keys()); N=len(code_list)
    if N<50: continue

    idx_pct=csi_map.get(d,0.0)
    pct_a=np.array([last_pct[c] for c in code_list],dtype=np.float32)
    amt_a=np.array([last_amt[c] for c in code_list],dtype=np.float32)
    tvr_a=np.array([last_tvr[c] for c in code_list],dtype=np.float32)
    pe_a=np.array([last_pe[c] for c in code_list],dtype=np.float32)
    pb_a=np.array([last_pb[c] for c in code_list],dtype=np.float32)
    cm_a=np.array([last_cm[c] for c in code_list],dtype=np.float32)
    cf=np.zeros((N,7),dtype=np.float32)
    cf[:,0]=rank_pct(pct_a,N); cf[:,1]=rank_pct(amt_a,N); cf[:,2]=rank_pct(tvr_a,N)
    cf[:,3]=pct_a-np.float32(idx_pct)
    cf[:,4]=rank_pct(np.clip(pe_a,*PE_CLIP),N); cf[:,5]=rank_pct(np.clip(pb_a,*PB_CLIP),N); cf[:,6]=rank_pct(cm_a,N)
    cn=norm_c(cf)

    ta=[]
    for ts_code in code_list:
        vr,vg=windows[ts_code]; ta.append(np.concatenate([vr,vg[:,None]],axis=1))
    ts_=np.stack(ta,0); Ns,Ts,Fs=ts_.shape
    tf=ts_.reshape(-1,Fs); tf=winsorize_2d(tf); ts_=tf.reshape(Ns,Ts,Fs)

    batch=[]
    for i in range(N):
        t=norm_t(ts_[i]); ct=np.tile(cn[i],(W,1)); batch.append(np.concatenate([t,ct],-1))

    feat=np.stack(batch)
    t=torch.from_numpy(feat).float().to(device)
    with torch.no_grad(): scores=m(t).cpu().numpy()
    for i,c in enumerate(code_list): records.append({"trade_date":d,"ts_code":c,"score":float(scores[i])})
    print(f"  [{di+1}/{len(test_dates)}] {d} N={N}")

new_df=pd.DataFrame(records)
existing=pd.read_parquet(OUT)
# Filter out existing dates to avoid duplicates
existing_dates=set(existing["trade_date"].unique())
new_df=new_df[~new_df["trade_date"].isin(existing_dates)]
merged=pd.concat([existing,new_df],ignore_index=True)
merged.to_parquet(OUT,index=False)
print(f"[extend] Done. {len(existing)} → {len(merged)} records ({len(new_df)} new)")
