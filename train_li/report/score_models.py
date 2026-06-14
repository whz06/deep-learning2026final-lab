"""report/score_models.py — Unified batch scoring (v2: groupby + vectorized).

Usage (Windows PowerShell):
  & D:\Software\miniconda3\envs\dl_lab1\python.exe D:\...\report\score_models.py --model v6    --start 20260105 --end 20260529
  & D:\Software\miniconda3\envs\dl_lab1\python.exe D:\...\report\score_models.py --model v7gru --start 20260105 --end 20260529
  & D:\Software\miniconda3\envs\dl_lab1\python.exe D:\...\report\score_models.py --model v8    --start 20260105 --end 20260529
  & D:\Software\miniconda3\envs\dl_lab1\python.exe D:\...\report\score_models.py --model v7spatial --start 20260601 --end 20260612
"""
import os, sys, argparse, glob, time, numpy as np, pandas as pd, torch, torch.nn as nn, torch.nn.functional as F

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(SCRIPT_DIR)

# Use ORIGINAL model imports (not inlined) to avoid subtle architecture mismatches
sys.path.insert(0, ROOT)
from v7.models.gru_spatial import GRURankerSpatial as GRURankerSpatialV7
from v7.models.spatial_attn import SparseSpatialAttention as SparseSpatialAttentionV7
from v2.models.gru import GRURanker as GRURankerV2

# V8 needs its own import
from v8.models.gru_spatial_v8 import GRURankerSpatialV8 as GRURankerSpatialV8_orig
from v8.models.spatial_attn_v8 import SparseSpatialAttention as SparseSpatialAttentionV8_orig

MODELS = {
    "v6":       {"ckpt":"v6/checkpoints/gru_spatial_concat_d32_K5_H128_L1_D0.2_lr0.0003_N1024.pt",
                 "out":"v6/results/daily_scores.parquet","dim":26,"spatial":"v6"},
    "v7gru":    {"ckpt":"v7/checkpoints/gru_v7_H128_L1_D0.2_lr0.0003_N2048.pt",
                 "out":"v7/results/daily_scores_gru_t1.parquet","dim":26,"spatial":None},
    "v7spatial":{"ckpt":"v7/checkpoints/gru_spatial_v7_d32_K5_H128_L1_D0.2_lr0.0003_N1024.pt",
                 "out":"v7/results/daily_scores_spatial_t1.parquet","dim":26,"spatial":"v7"},
    "v8":       {"ckpt":"v8/checkpoints/v8_listmle_d32_K5_H128_L1_D0.2_lr0.0003_N1024_ind111.pt",
                 "out":"v8/results/daily_scores_v8.parquet","dim":31,"spatial":"v8",
                 "needs_moneyflow":True,"needs_industry":True},
}

W, WARMUP = 60, 300
RAW = ["open","high","low","close","vol","amount","pct_chg","turnover_rate","volume_ratio","total_mv"]
TECH = ["macd","macd_signal","rsi","bb_width","bb_pct","mom_5","mom_20","vol_20"]
V8_TIER1 = ["amihud_20","price_pos_20","ret_skew_20"]
LOG_COLS = ["vol","amount","total_mv"]
TDIM_V67 = len(RAW)+len(TECH)+1  # 19
TDIM_V8  = len(RAW)+len(TECH)+len(V8_TIER1)+1  # 22

# ═══════════ Model Classes ═══════════
class GRURanker(nn.Module):
    def __init__(self,idim,hs=128,nl=1,do=0.2):
        super().__init__()
        self.gru=nn.GRU(idim,hs,nl,batch_first=True,dropout=do if nl>1 else 0.0)
        self.head=nn.Sequential(nn.Linear(hs,hs//2),nn.ReLU(),nn.Dropout(do),nn.Linear(hs//2,1))
    def forward(self,x): out,_=self.gru(x); return self.head(out[:,-1,:]).squeeze(-1)

class SparseSpatialAttention(nn.Module):
    def __init__(self,dm=128,dp=32,K=5):
        super().__init__()
        self.query=nn.Linear(dm,dp,bias=False); self.key=nn.Linear(dm,dp,bias=False)
        self.value=nn.Linear(dm,dp,bias=False); self.K,self.scale=K,dp**0.5
    def forward(self,h):
        N=h.size(0); q,k,v=self.query(h),self.key(h),self.value(h)
        sim=q@k.T/self.scale; sim.fill_diagonal_(-float('inf'))
        Ke=min(self.K,N-1); ts,ti=sim.topk(Ke,dim=-1)
        a=F.softmax(ts,dim=-1); return (a.unsqueeze(1)@v[ti]).squeeze(1)

class GRURankerSpatial(GRURanker):
    def __init__(self,idim,hs=128,nl=1,do=0.2,dp=32,K=5):
        super().__init__(idim,hs,nl,do)
        self.spatial=SparseSpatialAttention(hs,dp,K)
        self.head=nn.Sequential(nn.Linear(hs+dp,hs//2),nn.ReLU(),nn.Dropout(do),nn.Linear(hs//2,1))
    def forward(self,x):
        out,_=self.gru(x); h=out[:,-1,:]; ctx=self.spatial(h)
        return self.head(torch.cat([h,ctx],dim=-1)).squeeze(-1)

class SparseSpatialAttentionV8(nn.Module):
    def __init__(self,dm=128,dp=32,K=5,lg=0.1):
        super().__init__()
        self.query=nn.Linear(dm,dp,bias=False); self.key=nn.Linear(dm,dp,bias=False)
        self.value=nn.Linear(dm,dp,bias=False); self.K,self.scale,self.lg=K,dp**0.5,lg
    def forward(self,h,ind_ids=None):
        N=h.size(0); q,k,v=self.query(h),self.key(h),self.value(h)
        sim=q@k.T/self.scale; sim.fill_diagonal_(-float('inf'))
        if ind_ids is not None: sim=sim+self.lg*(ind_ids.unsqueeze(0)==ind_ids.unsqueeze(1)).float()
        Ke=min(self.K,N-1); ts,ti=sim.topk(Ke,dim=-1)
        a=F.softmax(ts,dim=-1); return (a.unsqueeze(1)@v[ti]).squeeze(1)

class GRURankerSpatialV8(GRURanker):
    def __init__(self,idim,hs=128,nl=1,do=0.2,dp=32,K=5,ni=111,ied=8,lg=0.1):
        super().__init__(idim,hs,nl,do)
        self.spatial=SparseSpatialAttentionV8(hs,dp,K,lg)
        self.ind_emb=nn.Embedding(ni,ied)
        self.head=nn.Sequential(nn.Linear(hs+dp+ied,hs//2),nn.ReLU(),nn.Dropout(do),nn.Linear(hs//2,1))
    def forward(self,x,ind_ids=None):
        out,_=self.gru(x); h=out[:,-1,:]; ctx=self.spatial(h,ind_ids)
        ie=self.ind_emb(ind_ids) if ind_ids is not None else torch.zeros(h.size(0),8,device=h.device)
        return self.head(torch.cat([h,ctx,ie],dim=-1)).squeeze(-1)

# ═══════════ Feature Engineering ═══════════
def add_tech(df):
    c=df["close"].astype(float)
    e12,e26=c.ewm(span=12,adjust=False).mean(),c.ewm(span=26,adjust=False).mean()
    df["macd"]=e12-e26; df["macd_signal"]=df["macd"].ewm(span=9,adjust=False).mean()
    d=c.diff(); g=d.clip(lower=0); l=(-d).clip(lower=0)
    rs=g.ewm(alpha=1/14,adjust=False).mean()/(l.ewm(alpha=1/14,adjust=False).mean()+1e-8)
    df["rsi"]=100-100/(1+rs)
    m20=c.rolling(20).mean(); s20=c.rolling(20).std()
    df["bb_width"]=2*s20/(m20+1e-8); df["bb_pct"]=(c-(m20-2*s20))/(4*s20+1e-8)
    df["mom_5"]=c/c.shift(5)-1; df["mom_20"]=c/c.shift(20)-1
    df["vol_20"]=c.pct_change().rolling(20).std()
    return df

def add_v8_features(df):
    ret=df["pct_chg"].astype(float)/100.0; amt=df["amount"].astype(float)
    df["amihud_20"]=(np.abs(ret)/np.maximum(amt,1e-8)).rolling(20,min_periods=5).mean()
    c=df["close"].astype(float)
    L20=c.rolling(20,min_periods=5).min(); H20=c.rolling(20,min_periods=5).max()
    df["price_pos_20"]=(c-L20)/np.maximum(H20-L20,1e-8)
    df["ret_skew_20"]=ret.rolling(20,min_periods=10).skew().fillna(0)
    return df

def winsorize(arr):
    lo=np.nanpercentile(arr,1,axis=0,keepdims=True); hi=np.nanpercentile(arr,99,axis=0,keepdims=True)
    return np.clip(arr,lo,hi)
def norm_t(arr):
    m=np.nanmean(arr,axis=0,keepdims=True); s=np.nanstd(arr,axis=0,keepdims=True)+1e-8; return (arr-m)/s
def norm_c(arr):
    m=np.nanmean(arr,axis=0,keepdims=True); s=np.nanstd(arr,axis=0,keepdims=True)+1e-8; return (arr-m)/s
def rank_pct(arr):
    N=len(arr); valid=~np.isnan(arr); out=np.full(N,0.5,dtype=np.float32)
    if valid.sum()>=2:
        order=np.argsort(np.argsort(arr[valid])); out[valid]=order.astype(np.float32)/max(valid.sum()-1,1)
    return out

def compute_mf(ts_arr,mf_lookup,date):
    N=len(ts_arr); hhi=np.zeros(N,dtype=np.float32); sm_lg=np.zeros(N,dtype=np.float32)
    if date not in mf_lookup: return hhi,sm_lg
    mf_day=mf_lookup[date]
    idx_df=pd.DataFrame({"ts_code":ts_arr,"order":range(N)}).set_index("ts_code")
    common=mf_day.index.intersection(idx_df.index)
    if len(common)==0: return hhi,sm_lg
    sub=mf_day.loc[common]; orders=idx_df.loc[common,"order"].values
    cols=[c for c in sub.columns if c.endswith("_vol")]
    if cols:
        vols=sub[cols].abs().fillna(0).values; totals=vols.sum(axis=1,keepdims=True)+1e-8
        hhi[orders]=((vols/totals)**2).sum(axis=1).astype(np.float32)
    if "buy_sm_vol" in sub.columns and "sell_lg_vol" in sub.columns:
        sm_lg[orders]=(sub["buy_sm_vol"].fillna(0)-sub["sell_lg_vol"].fillna(0)).values.astype(np.float32)
    return hhi,sm_lg

# ═══════════ Main ═══════════
def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--model",required=True,choices=list(MODELS.keys()))
    parser.add_argument("--start",required=True); parser.add_argument("--end",required=True)
    parser.add_argument("--device",default="cuda"); parser.add_argument("--batch-size",type=int,default=4096)
    args=parser.parse_args()

    cfg=MODELS[args.model]
    ckpt_path=os.path.join(ROOT,cfg["ckpt"]); out_path=os.path.join(ROOT,cfg["out"])
    parquet_path=os.path.join(ROOT,"processed","all_data.parquet")
    idx_path=os.path.join(ROOT,"data","market","000300.SH.csv")
    basic_path=os.path.join(ROOT,"data","basic.csv")
    mf_dir=os.path.join(ROOT,"data","moneyflow")

    device=torch.device(args.device if torch.cuda.is_available() else "cpu")
    is_v8=cfg.get("needs_moneyflow",False)
    print(f"[{args.model}] device={device} dim={cfg['dim']} spatial={cfg['spatial']} {args.start}→{args.end}")

    # ── Model (use original imports for V7/V8 to avoid inlined-class bugs) ──
    dim,st=cfg["dim"],cfg.get("spatial")
    if st=="v6":    model=GRURankerSpatial(dim,128,1,0.2,32,5)
    elif st=="v7":  model=GRURankerSpatialV7(input_dim=dim,hidden_size=128,num_layers=1,dropout=0.2,d_proj=32,K=5)
    elif st=="v8":  model=GRURankerSpatialV8_orig(input_dim=dim,hidden_size=128,num_layers=1,dropout=0.2,d_proj=32,K=5,n_industries=111,ind_emb_dim=8,lambda_gate=0.1)
    else:           model=GRURankerV2(input_dim=dim,hidden_size=128,num_layers=1,dropout=0.2)
    state=torch.load(ckpt_path,map_location="cpu",weights_only=True)
    state={k.replace("_orig_mod.",""):v for k,v in state.items()}
    model.load_state_dict(state,strict=True); model.to(device).eval()
    # Warmup forward to initialize CUDA kernels (fixes first-batch NaN with spatial attention)
    with torch.no_grad():
        if is_v8 and cfg.get("needs_industry"):
            _ = model(torch.randn(64,W,dim,device=device), torch.randint(0,111,(64,),device=device))
        else:
            _ = model(torch.randn(64,W,dim,device=device))
    torch.cuda.synchronize()
    print(f"[{args.model}] params={sum(p.numel() for p in model.parameters()):,}")

    # ── Data ──
    t0=time.time()
    df=pd.read_parquet(parquet_path); df["trade_date"]=df["trade_date"].astype(str)
    all_dates=sorted(df["trade_date"].unique())
    if args.start not in all_dates: print(f"ERROR: {args.start} not found"); sys.exit(1)
    end_date=args.end if args.end in all_dates else all_dates[-1]
    warmup_cutoff=max(0,all_dates.index(args.start)-WARMUP)
    df=df[df["trade_date"]>=all_dates[warmup_cutoff]]
    print(f"[*] Data: {df['trade_date'].min()}~{df['trade_date'].max()} ({time.time()-t0:.1f}s)")

    # ── CSI300 ──
    csi=pd.read_csv(idx_path,dtype={"trade_date":str})
    csi_map=dict(zip(csi["trade_date"],csi["pct_chg"].astype(np.float32)))

    # ── Industry ──
    ts2ind={}
    if cfg.get("needs_industry"):
        basic=pd.read_csv(basic_path,dtype={"ts_code":str})
        basic["industry"]=basic["industry"].fillna("Other")
        ind_list=sorted(basic["industry"].unique()); ind2id={ind:i for i,ind in enumerate(ind_list)}
        ts2ind=dict(zip(basic["ts_code"],basic["industry"].map(ind2id)))

    # ── Moneyflow ──
    mf_lookup={}
    if is_v8:
        for f in sorted(glob.glob(os.path.join(mf_dir,"*.csv"))):
            d=os.path.basename(f).replace(".csv","")
            try:
                chunk=pd.read_csv(f,dtype={"ts_code":str,"trade_date":str})
                needed=[c for c in chunk.columns if c.endswith("_vol")]
                if needed: mf_lookup[d]=chunk.set_index("ts_code")[needed]
            except Exception: pass

    # ═══════════ Feature store: groupby (NOT df[df["ts_code"]==ts]) ═══════════
    print("[*] Building feature store (groupby) ..."); t0=time.time()

    # Pre-sort once
    df=df.sort_values(["ts_code","trade_date"]).reset_index(drop=True)

    temporal_list=[]; lastval_list=[]; date_pos_list=[]; stock_ids=[]
    tdims_per_stock=[]

    for ts,sdf in df.groupby("ts_code",sort=True):
        sdf=sdf.reset_index(drop=True)
        if len(sdf)<W: continue

        for col in RAW:
            if col in sdf.columns: sdf[col]=sdf[col].ffill()
        sdf=add_tech(sdf)
        if is_v8: sdf=add_v8_features(sdf)
        sdf["vwap_gap"]=(sdf["close"]/sdf["vwap"]-1) if "vwap" in sdf.columns else 0.0

        raw_a=sdf[RAW].astype(np.float32).values
        tech_a=sdf[TECH].astype(np.float32).values
        vwap_a=sdf["vwap_gap"].astype(np.float32).values.reshape(-1,1)
        if is_v8:
            tier1_a=sdf[V8_TIER1].astype(np.float32).values
            temp=np.concatenate([raw_a,tech_a,tier1_a,vwap_a],axis=1)
        else:
            temp=np.concatenate([raw_a,tech_a,vwap_a],axis=1)
        for ci,col in enumerate(RAW):
            if col in LOG_COLS: temp[:,ci]=np.log1p(np.maximum(temp[:,ci],0))
        temporal_list.append(temp)
        tdims_per_stock.append(len(temp))

        lv=np.column_stack([
            sdf["pct_chg"].astype(np.float32).values,
            sdf["amount"].astype(np.float32).values,
            sdf["turnover_rate"].astype(np.float32).values if "turnover_rate" in sdf.columns else np.zeros(len(sdf),dtype=np.float32),
            sdf["pe"].astype(np.float32).values if "pe" in sdf.columns else np.zeros(len(sdf),dtype=np.float32),
            sdf["pb"].astype(np.float32).values if "pb" in sdf.columns else np.zeros(len(sdf),dtype=np.float32),
            sdf["circ_mv"].astype(np.float32).values if "circ_mv" in sdf.columns else np.zeros(len(sdf),dtype=np.float32),
        ])
        lastval_list.append(lv)

        dates=sdf["trade_date"].tolist()
        date_pos_list.append({d:i for i,d in enumerate(dates)})
        stock_ids.append(ts)

    n_stocks=len(stock_ids)
    # Pre-compute stock_id→index mapping for O(1) access
    stock_to_idx={ts:i for i,ts in enumerate(stock_ids)}

    # Also pre-build ind_id array for V8
    ind_arr=None
    if is_v8 and cfg.get("needs_industry"):
        ind_arr=np.array([ts2ind.get(ts,0) for ts in stock_ids],dtype=np.int64)

    print(f"[*] Feature store: {n_stocks} stocks in {time.time()-t0:.1f}s")

    # ── Target dates ──
    target_dates=[d for d in all_dates if args.start<=d<=end_date]
    existing=None
    if os.path.exists(out_path):
        existing=pd.read_parquet(out_path)
        ed=set(existing["trade_date"].unique())
        target_dates=[d for d in target_dates if d not in ed]
        print(f"[*] existing={len(ed)} new={len(target_dates)}")
    else:
        print(f"[*] target={len(target_dates)} dates")
    if not target_dates: print("[*] All done."); return

    # ═══════════ Score each date ═══════════
    all_rows=[]
    bs=min(args.batch_size,n_stocks)

    for di,date in enumerate(target_dates):
        t_day=time.time()

        # Find valid stocks + positions (single pass, O(1) lookup)
        valid_idx=[]; positions=[]
        for i in range(n_stocks):
            pos=date_pos_list[i].get(date)
            if pos is not None and pos>=W-1:
                valid_idx.append(i); positions.append(pos)
        N=len(valid_idx)
        if N<10: continue
        valid_idx=np.array(valid_idx,dtype=np.int32)
        positions=np.array(positions,dtype=np.int32)

        # Extract windows + lastvals via numpy indexing (NO per-stock loop)
        tdim = TDIM_V8 if is_v8 else TDIM_V67
        temporal_wins=np.zeros((N,W,tdim),dtype=np.float32)
        last_vals=np.zeros((N,6),dtype=np.float32)
        for j,(si,pos) in enumerate(zip(valid_idx,positions)):
            temporal_wins[j]=temporal_list[si][pos-W+1:pos+1]
            last_vals[j]=lastval_list[si][pos]

        temporal_wins=norm_t(winsorize(temporal_wins))

        # Cross-sectional
        pct_a=last_vals[:,0]; amt_a=last_vals[:,1]; to_a=last_vals[:,2]
        idx_pct=np.float32(csi_map.get(date,0.0))
        cross=np.stack([rank_pct(pct_a),rank_pct(amt_a),rank_pct(to_a),pct_a-idx_pct],axis=1)
        cross=norm_c(cross); cross_t=np.tile(cross[:,np.newaxis,:],(1,W,1))

        # Valuation
        val=np.stack([rank_pct(last_vals[:,3]),rank_pct(last_vals[:,4]),rank_pct(last_vals[:,5])],axis=1)
        val=norm_c(val); val_t=np.tile(val[:,np.newaxis,:],(1,W,1))

        # Feature tensor
        if is_v8:
            ts_arr=np.array([stock_ids[i] for i in valid_idx])
            hhi,smlg=compute_mf(ts_arr,mf_lookup,date)
            mf=norm_c(np.stack([hhi,smlg],axis=1))
            mf_t=np.tile(mf[:,np.newaxis,:],(1,W,1))
            feats=np.concatenate([temporal_wins,cross_t,val_t,mf_t],axis=2)
        else:
            feats=np.concatenate([temporal_wins,cross_t,val_t],axis=2)

        # GPU inference
        scores_all=[]
        for s in range(0,N,bs):
            e=min(s+bs,N); bx=torch.from_numpy(feats[s:e]).float().to(device,non_blocking=True)
            with torch.no_grad():
                if is_v8 and ind_arr is not None:
                    ids=torch.from_numpy(ind_arr[valid_idx[s:e]]).to(device)
                    sc=model(bx,ids)
                else:
                    sc=model(bx)
            scores_all.append(sc.cpu().numpy())
        scores=np.concatenate(scores_all)

        for j,si in enumerate(valid_idx):
            all_rows.append({"trade_date":date,"ts_code":stock_ids[si],"score":float(scores[j])})

        elapsed=time.time()-t_day
        if (di+1)%5==0 or di==0 or di==len(target_dates)-1:
            print(f"  [{di+1:>3}/{len(target_dates)}] {date}: {N} stk  μ={scores.mean():.3f} σ={scores.std():.3f}  {elapsed:.1f}s")

    # ── Save ──
    new_df=pd.DataFrame(all_rows)
    print(f"\n[*] new: {len(new_df):,} rows × {new_df['trade_date'].nunique()} dates")
    if existing is not None and len(existing)>0:
        existing=existing[~existing["trade_date"].isin(new_df["trade_date"].unique())]
        final=pd.concat([existing,new_df],ignore_index=True)
        print(f"[*] merged: {len(final):,} rows")
    else:
        final=new_df
    final.to_parquet(out_path,index=False)
    print(f"[*] saved → {out_path}")
    print(f"[{args.model}] Done.")

if __name__=="__main__":
    main()
