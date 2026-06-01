"""v7/validate.py — Precompute + IC + 10x10 backtest for both v7 models.

1. Precompute scores on 600 sampled stocks (Feb-May 2026)
2. Compute T+1 IC stats
3. Run 10x10 backtest + sweep vs v3 baseline (+6.85% cum, Sharpe 0.93)
"""
import os, sys, gc, numpy as np, pandas as pd, torch, torch.nn as nn
from scipy.stats import spearmanr

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(SCRIPT_DIR)
PARQUET = os.path.join(ROOT, "processed", "all_data.parquet")
INDEX_P = os.path.join(ROOT, "data", "market", "000300.SH.csv")

# Models
sys.path.insert(0, os.path.join(ROOT, "v2"))
from models.gru import GRURanker

sys.path.insert(0, ROOT)
from v7.models.gru_spatial import GRURankerSpatial

W, START, END = 60, "20260203", "20260529"
N_SAMPLED = 600

RAW = ["open","high","low","close","vol","amount","pct_chg","turnover_rate","volume_ratio","total_mv"]
TECH = ["macd","macd_signal","rsi","bb_width","bb_pct","mom_5","mom_20","vol_20"]
LOG_COLS = ["vol","amount","total_mv"]
PE_CLIP, PB_CLIP = (0.1,500.0), (0.1,50.0)
WINSOR_P = (1,99)

CSI5D_THRESH = -1.0; RISK_OFF = 0.80

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

def winsorize_2d(arr, p_low=1, p_high=99):
    lo=np.percentile(arr,p_low,axis=0,keepdims=True); hi=np.percentile(arr,p_high,axis=0,keepdims=True)
    return np.clip(arr,lo,hi)

def norm_t(arr): m,s=arr.mean(axis=0,keepdims=True),arr.std(axis=0,keepdims=True)+1e-8; return (arr-m)/s
def norm_c(arr): m,s=arr.mean(axis=0,keepdims=True),arr.std(axis=0,keepdims=True)+1e-8; return (arr-m)/s

def rank_pct(arr, N):
    valid=~np.isnan(arr); out=np.zeros(N,dtype=np.float32)
    if valid.sum()>=2: order=np.argsort(np.argsort(arr[valid])); out[valid]=order.astype(np.float32)/max(valid.sum()-1,1)
    return out

# ========== Load data ==========
print("[validate] Loading data ...")
np.random.seed(42)
scores_all = pd.read_parquet(os.path.join(ROOT, "v6", "results", "daily_scores.parquet"))
dates_s = sorted(scores_all["trade_date"].unique())

# Load data with 300-day warmup before test period
df = pd.read_parquet(PARQUET)
df["trade_date"] = df["trade_date"].astype(str)
all_dates = sorted(df["trade_date"].unique())
test_start_idx = all_dates.index(START) if START in all_dates else 0
warmup_cutoff = max(0, test_start_idx - 300)
df = df[df["trade_date"] >= all_dates[warmup_cutoff]]
all_stocks = sorted(df["ts_code"].unique())
sampled = sorted(np.random.choice(all_stocks, N_SAMPLED, replace=False))
df = df[df["ts_code"].isin(sampled)]
print(f"[validate] Data: {df['trade_date'].min()} ~ {df['trade_date'].max()}, {df['ts_code'].nunique()} stocks, {len(df):,} rows")

# Build returns & MV lookup (for dates in our data range)
ret_d, mv_d = {}, {}
for d in sorted(set(df["trade_date"])):
    sub = df[df["trade_date"]==d]
    ret_d[d] = dict(zip(sub["ts_code"], sub["pct_chg"].astype(float)))
    mv_d[d] = dict(zip(sub["ts_code"], sub["total_mv"].astype(float)))

csi = pd.read_csv(INDEX_P, dtype={"trade_date":str})
csi_map = dict(zip(csi["trade_date"], csi["pct_chg"].astype(float)))
csi_dates = sorted(csi_map.keys())

def get_csi5d(date):
    if date not in csi_dates: return 0.0
    idx=csi_dates.index(date); start=max(0,idx-4)
    return sum(csi_map[csi_dates[i]] for i in range(start,idx+1))

# Pre-load stock series for feature computation
print("[validate] Pre-loading stock series ...")
series = {}
for ts, sdf in df.groupby("ts_code"):
    sdf = sdf.sort_values("trade_date").reset_index(drop=True).copy()
    sdf = sdf.ffill()
    series[ts] = sdf  # no add_tech here, do it on the fly

print(f"[validate] Series loaded: {len(series)} stocks")

# ========== Feature computation ==========
def compute_features(tgt_date, idx_map):
    """Compute 26-dim features for all stocks at tgt_date. Returns (features[N,T,26], codes list)."""
    windows, last_pct, last_amt, last_tvr, last_pe, last_pb, last_cm = {}, {}, {}, {}, {}, {}, {}
    
    for ts, sdf in series.items():
        sdf_c = sdf[sdf["trade_date"] <= tgt_date]
        if len(sdf_c) < W: continue
        sdf_c = sdf_c.ffill()
        sdf_c = add_tech(sdf_c.copy())
        
        vals_raw = sdf_c[RAW + TECH].values.astype(np.float32)[-W:]
        for cn in LOG_COLS:
            if cn in RAW:
                vals_raw[:, RAW.index(cn)] = np.log1p(np.maximum(vals_raw[:, RAW.index(cn)], 0))
        
        wc = sdf_c["close"].values.astype(np.float32)[-W:]
        wv = sdf_c["vwap"].values.astype(np.float32)[-W:]
        vg = wc / np.maximum(wv, 1e-8) - 1
        
        if np.isnan(vals_raw).any() or np.isnan(vg).any(): continue
        
        windows[ts] = (vals_raw, vg)
        lr = sdf_c.iloc[-1]
        last_pct[ts] = lr["pct_chg"]; last_amt[ts] = lr["amount"]; last_tvr[ts] = lr["turnover_rate"]
        last_pe[ts] = lr.get("pe",0); last_pb[ts] = lr.get("pb",0); last_cm[ts] = lr.get("circ_mv",0)
    
    code_list = list(windows.keys()); N = len(code_list)
    if N < 50: return None, None
    
    idx_pct = idx_map.get(tgt_date, 0.0)
    pct_a = np.array([last_pct[c] for c in code_list], dtype=np.float32)
    amt_a = np.array([last_amt[c] for c in code_list], dtype=np.float32)
    tvr_a = np.array([last_tvr[c] for c in code_list], dtype=np.float32)
    pe_a  = np.array([last_pe[c] for c in code_list], dtype=np.float32)
    pb_a  = np.array([last_pb[c] for c in code_list], dtype=np.float32)
    cm_a  = np.array([last_cm[c] for c in code_list], dtype=np.float32)
    
    cf = np.zeros((N, 7), dtype=np.float32)
    cf[:,0] = rank_pct(pct_a, N); cf[:,1] = rank_pct(amt_a, N); cf[:,2] = rank_pct(tvr_a, N)
    cf[:,3] = pct_a - np.float32(idx_pct)
    cf[:,4] = rank_pct(np.clip(pe_a, *PE_CLIP), N); cf[:,5] = rank_pct(np.clip(pb_a, *PB_CLIP), N)
    cf[:,6] = rank_pct(cm_a, N)
    cn = norm_c(cf)
    
    ta = []
    for ts_code in code_list:
        vr, vg = windows[ts_code]
        ta.append(np.concatenate([vr, vg[:,None]], axis=1))
    ts_ = np.stack(ta, 0)
    Ns, Ts, Fs = ts_.shape
    tf = ts_.reshape(-1, Fs); tf = winsorize_2d(tf, *WINSOR_P); ts_ = tf.reshape(Ns, Ts, Fs)
    
    batch = []
    for i in range(N):
        t = norm_t(ts_[i])
        ct = np.tile(cn[i], (W, 1))
        batch.append(np.concatenate([t, ct], -1))
    
    return np.stack(batch), code_list

# ========== Load both models ==========
device = torch.device("cuda")
def load_stripped(model, path, device):
    state = torch.load(path, map_location=device, weights_only=True)
    state = {k.replace("_orig_mod.", ""): v for k, v in state.items()}
    model.load_state_dict(state)

print("[validate] Loading v7 GRU ...")
m_gru = GRURanker(26,128,1,0.2).to(device).eval()
load_stripped(m_gru, os.path.join(SCRIPT_DIR,"checkpoints","gru_v7_H128_L1_D0.2_lr0.0003_N2048.pt"), device)

print("[validate] Loading v7 Spatial ...")
m_spatial = GRURankerSpatial(26,128,1,0.2,d_proj=32,K=5).to(device).eval()
load_stripped(m_spatial, os.path.join(SCRIPT_DIR,"checkpoints","gru_spatial_v7_d32_K5_H128_L1_D0.2_lr0.0003_N1024.pt"), device)

# ========== Precompute scores ==========
print(f"\n[validate] Scoring {len(dates_s)} test dates for both models ...")
test_dates = [d for d in dates_s if START <= d <= END]
score_records = {"gru": [], "spatial": []}

for di, d in enumerate(test_dates):
    feat, codes = compute_features(d, csi_map)
    if feat is None:
        if di < 2: print(f"  [DEBUG] {d}: compute_features returned None")
        continue
    if di < 2: print(f"  [DEBUG] {d}: got {feat.shape} features for {len(codes)} stocks")
    t = torch.from_numpy(feat).float().to(device)
    with torch.no_grad():
        sg = m_gru(t).cpu().numpy()
        ss = m_spatial(t).cpu().numpy()
    for i, c in enumerate(codes):
        score_records["gru"].append({"trade_date":d,"ts_code":c,"score":float(sg[i])})
        score_records["spatial"].append({"trade_date":d,"ts_code":c,"score":float(ss[i])})
    if (di+1)%20==0: print(f"  [{di+1}/{len(test_dates)}] {d}")

pd.DataFrame(score_records["gru"]).to_parquet(os.path.join(SCRIPT_DIR,"results","daily_scores_gru_t1.parquet"), index=False)
pd.DataFrame(score_records["spatial"]).to_parquet(os.path.join(SCRIPT_DIR,"results","daily_scores_spatial_t1.parquet"), index=False)
print(f"[validate] Scores saved.")

# ========== IC analysis ==========
print(f"\n{'='*60}")
print(" IC Analysis (T+1) on test period")
print(f"{'='*60}")

for model_name, recs in [("v7 GRU", score_records["gru"]), ("v7 Spatial", score_records["spatial"])]:
    sd = {}
    for r in recs: sd.setdefault(r["trade_date"], {})[r["ts_code"]] = r["score"]
    
    ics = []
    for i in range(len(dates_s)-1):
        fd, rd = dates_s[i], dates_s[i+1]
        if fd not in sd or rd not in ret_d: continue
        ss = sd[fd]; rr = ret_d[rd]
        common = [ts for ts in ss if ts in rr]
        if len(common) < 50: continue
        s_val = [ss[ts] for ts in common]; r_val = [rr[ts] for ts in common]
        ic = spearmanr(s_val, r_val).correlation
        if not np.isnan(ic): ics.append(ic)
    
    print(f"  {model_name}: IC={np.mean(ics):+.4f} std={np.std(ics):.4f} >0={np.mean([x>0 for x in ics]):.1%} n={len(ics)}")

# ========== 10x10 backtest ==========
print(f"\n{'='*60}")
print(" 10x10 Random Backtest (with Strategy B)")
print(f"{'='*60}")

def backtest_10x10(sd_name, sd_records):
    sd = {}
    for r in sd_records: sd.setdefault(r["trade_date"], {})[r["ts_code"]] = r["score"]
    
    # Build returns for all dates
    test_list = sorted(sd.keys())
    
    np.random.seed(42)
    valid_starts = [d for d in test_list if test_list.index(d) < len(test_list)-10]
    starts = sorted(np.random.choice(valid_starts, 10, replace=False))
    
    results = []
    for wi, start in enumerate(starts):
        si = test_list.index(start)
        end = test_list[si+9]
        holdings = None
        daily_p, daily_c = [], []
        
        for t in range(10):
            fd = test_list[si+t]
            rd = test_list[si+t+1] if si+t+1 < len(test_list) else fd
            
            if fd not in sd or rd not in ret_d: continue
            ss = sd[fd]; rr = ret_d[rd]
            sorted_s = sorted(ss.items(), key=lambda x:x[1], reverse=True)
            top = [c for c,_ in sorted_s if c in rr]
            if len(top) < 20: continue
            
            csi5d = get_csi5d(fd)
            pos = RISK_OFF if csi5d < CSI5D_THRESH else 1.0
            target = max(1, int(round(20*pos)))
            
            if holdings is None:
                holdings = top[:target]
            else:
                held_s = [(c, ss.get(c,-1e6)) for c in holdings if c in ss]
                held_s.sort(key=lambda x:x[1], reverse=True)
                to_sell = {c for c,_ in held_s[-5:]} if len(held_s) > 5 else set()
                extra = len(holdings) - target
                if extra > 0:
                    cand = [c for c,_ in held_s if c not in to_sell]
                    for c in cand[-extra:]: to_sell.add(c)
                holdings = [c for c in holdings if c not in to_sell]
                held_set = set(holdings)
                for c in top:
                    if len(holdings) >= target: break
                    if c not in held_set: holdings.append(c)
            
            pr = np.mean([rr.get(c,0.0) for c in holdings]) if holdings else 0.0
            cr = csi_map.get(rd, 0.0)
            daily_p.append(pr); daily_c.append(cr)
        
        if daily_p:
            cum_p = np.sum(daily_p); cum_c = np.sum(daily_c)
            results.append({"window":wi+1,"start":start,"end":end,"cum":round(cum_p,4),"csi":round(cum_c,4)})
            print(f"  Win {wi+1}: {start}~{end} | port={cum_p:+.3f}% csi={cum_c:+.3f}%")
    
    if results:
        prs = [r["cum"] for r in results]; crs = [r["csi"] for r in results]
        exs = [p-c for p,c in zip(prs,crs)]
        wins = sum(e>0 for e in exs)
        print(f"  -> {wins}/10 wins, cum={np.mean(prs):+.3f}%, excess={np.mean(exs):+.3f}%")
    
    return results

r_gru = backtest_10x10("v7 GRU", score_records["gru"])
r_spatial = backtest_10x10("v7 Spatial", score_records["spatial"])

print(f"\n{'='*60}")
print(" Comparison vs v3 (v2 GRU T+1 + Strategy B)")
print(f"{'='*60}")
print(f"  v3 baseline: +6.85% cum, Sharpe 0.93, 6/10 wins")
for name, r in [("v7 GRU", r_gru), ("v7 Spatial", r_spatial)]:
    if r:
        prs=[x["cum"] for x in r]; crs=[x["csi"] for x in r]
        exs=[p-c for p,c in zip(prs,crs)]; wins=sum(e>0 for e in exs)
        print(f"  {name}: cum={np.mean(prs):+.3f}% excess={np.mean(exs):+.3f}% wins={wins}/10")

print(f"\n{'='*60}")
print(" Sweep: v7 Spatial (N_hold x sell_k) with Strategy B")
print(f"{'='*60}")

sd_sp = {}
for r in score_records["spatial"]:
    sd_sp.setdefault(r["trade_date"], {})[r["ts_code"]] = r["score"]

test_list = sorted(sd_sp.keys())
sweep_results = []

for N in [5,10,15,20,25,30]:
    for K in [2,3,5,8,10]:
        if K >= N: continue
        holdings = None
        daily_p, daily_c = [], []
        for si in range(len(test_list)-1):
            fd = test_list[si]; rd = test_list[si+1]
            if fd not in sd_sp or rd not in ret_d: continue
            ss = sd_sp[fd]; rr = ret_d[rd]
            sorted_s = sorted(ss.items(), key=lambda x:x[1], reverse=True)
            top = [c for c,_ in sorted_s if c in rr]
            if len(top) < N: continue
            
            csi5d = get_csi5d(fd)
            pos = RISK_OFF if csi5d < CSI5D_THRESH else 1.0
            target = max(1, int(round(N*pos)))
            
            if holdings is None:
                holdings = top[:target]
            else:
                held_s = [(c, ss.get(c,-1e6)) for c in holdings if c in ss]
                held_s.sort(key=lambda x:x[1], reverse=True)
                to_sell = {c for c,_ in held_s[-K:]} if len(held_s) > K else set()
                extra = len(holdings) - target
                if extra > 0:
                    cand = [c for c,_ in held_s if c not in to_sell]
                    for c in cand[-extra:]: to_sell.add(c)
                holdings = [c for c in holdings if c not in to_sell]
                held_set = set(holdings)
                for c in top:
                    if len(holdings) >= target: break
                    if c not in held_set: holdings.append(c)
            
            pr = np.mean([rr.get(c,0.0) for c in holdings]) if holdings else 0.0
            cr = csi_map.get(rd, 0.0)
            daily_p.append(pr); daily_c.append(cr)
        
        if daily_p:
            cum = np.sum(daily_p); cum_c = np.sum(daily_c)
            sweep_results.append({"N":N,"K":K,"cum":round(cum,4),"excess":round(cum-cum_c,4),
                                   "days":len(daily_p)})

sweep_df = pd.DataFrame(sweep_results).sort_values("cum", ascending=False)
print(f"{'N':>4} {'K':>4} {'Cum':>8} {'Excess':>8} {'Days':>5}")
for _, r in sweep_df.head(10).iterrows():
    print(f"{r['N']:>4.0f} {r['K']:>4.0f} {r['cum']:>+7.2f}% {r['excess']:>+7.2f}% {r['days']:>5}")
best = sweep_df.iloc[0]
print(f"\n  BEST: N={best['N']}, K={best['K']}: cum={best['cum']:+.2f}%")
