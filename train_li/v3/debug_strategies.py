"""Quick analysis: why C/A+C underperform baseline."""
import os, sys, numpy as np, pandas as pd, torch

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(SCRIPT_DIR)
V2_DIR = os.path.join(ROOT, "v2")
PARQUET_PATH = os.path.join(ROOT, "processed", "all_data.parquet")
INDEX_PATH   = os.path.join(ROOT, "data", "market", "000300.SH.csv")
CKPT_DIR = os.path.join(V2_DIR, "checkpoints")

sys.path.insert(0, V2_DIR)
from models.gru import GRURanker

WINDOW_SIZE = 60
RAW = ["open","high","low","close","vol","amount","pct_chg","turnover_rate","volume_ratio","total_mv"]
TECH = ["macd","macd_signal","rsi","bb_width","bb_pct","mom_5","mom_20","vol_20"]
CKPT = "gru_gru_hidden_size=128_num_layers=1_dropout=0.2_lr=0.0003.pt"

def add_tech(df):
    c = df["close"].astype(float)
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
    feats, codes, vol20s = [], [], []
    for ts, sdf in series.items():
        sd = sdf[sdf["trade_date"] <= date]
        if len(sd) < WINDOW_SIZE+1: continue
        sw = sd.iloc[-WINDOW_SIZE-1:]
        rv = sw[RAW + TECH].values.astype(np.float32)
        if np.isnan(rv).any(): continue
        feats.append(rv[-WINDOW_SIZE:]); codes.append(ts)
        vol20s.append(sw["vol_20"].values[-1])
    if not feats: return None, None, None, None
    fa = np.stack(feats, 0); n = len(feats)
    pv=fa[:,-1,6]; av=fa[:,-1,5]; tv=fa[:,-1,7]
    pr=np.argsort(np.argsort(pv)).astype(np.float32)/max(n-1,1)
    ar=np.argsort(np.argsort(av)).astype(np.float32)/max(n-1,1)
    tr=np.argsort(np.argsort(tv)).astype(np.float32)/max(n-1,1)
    ip=idx_map.get(date,0.0); rb=np.full(n,pv-ip,dtype=np.float32)
    cr=np.stack([np.tile(pr[:,None],(1,WINDOW_SIZE)),np.tile(ar[:,None],(1,WINDOW_SIZE)),
                  np.tile(tr[:,None],(1,WINDOW_SIZE)),np.tile(rb[:,None],(1,WINDOW_SIZE))],-1)
    full=np.concatenate([fa,cr],-1)
    m,s=full.mean(axis=0,keepdims=True),full.std(axis=0,keepdims=True)+1e-8
    return (full-m)/s, codes, pv, np.array(vol20s)


def main():
    device = torch.device("cuda")
    m = GRURanker(22,128,1,0.2)
    m.load_state_dict(torch.load(os.path.join(CKPT_DIR, CKPT), map_location=device, weights_only=True))
    m.to(device).eval()

    df = pd.read_parquet(PARQUET_PATH); df["trade_date"] = df["trade_date"].astype(str)
    dates_all = sorted(df["trade_date"].unique())
    tc = [d for d in dates_all if d >= "20260201"]
    lb = max(0, dates_all.index(tc[0]) - WINDOW_SIZE - 1 - 20)
    df = df[df["trade_date"] >= dates_all[lb]]
    idx_map = dict(zip(pd.read_csv(INDEX_PATH, dtype={"trade_date":str})["trade_date"],
                       pd.read_csv(INDEX_PATH, dtype={"trade_date":str})["pct_chg"]))

    stocks = sorted(df["ts_code"].unique())
    series = {}
    for ts in stocks:
        series[ts] = add_tech(df[df["ts_code"]==ts].sort_values("trade_date").reset_index(drop=True))

    test_dates = [d for d in dates_all if "20260201" <= d <= "20260531"]

    # Pre-compute scores for all dates
    print("Computing scores...")
    day_data = {}
    for di, date in enumerate(test_dates):
        if di == len(test_dates)-1: break
        nd = test_dates[di+1]
        features, codes, pct_now, vol20_arr = build(date, series, idx_map)
        if features is None: continue
        scores = m(torch.from_numpy(features).to(device)).detach().cpu().numpy()
        vl, vi = [], []
        for ci, c in enumerate(codes):
            r = series[c][series[c]["trade_date"]==nd]
            if len(r)>0: vl.append(r["pct_chg"].values[0]); vi.append(ci)
        if not vi: continue
        day_data[date] = {"scores":scores[vi],"codes":[codes[i] for i in vi],
                          "labels":np.array(vl),"vol_20":vol20_arr[vi]}
        torch.cuda.empty_cache()

    # CSI300 5-day
    csi_dates = sorted(day_data.keys())
    csi_5d = {}
    for i, d in enumerate(csi_dates):
        if i >= 5:
            rets = [idx_map.get(csi_dates[j], 0) for j in range(i-4, i+1)]
            csi_5d[d] = float(np.prod(1+np.array(rets)/100)-1)*100
        else:
            csi_5d[d] = 0.0

    print(f"\n{'Day':<12} {'CSI5d':>7} {'State':>8} {'All%':>7} "
          f"{'Top20%':>7} {'LoVol20%':>7} {'LoVol50%':>7} {'AlphaHi%':>7} {'GRU-IC':>7}")
    print("-"*85)

    for date in csi_dates:
        dd = day_data[date]
        csi5 = csi_5d[date]
        state = "BULL" if csi5>1 else ("BEAR" if csi5<-2 else "NEUTRAL")
        lv = dd["labels"]; vol20 = dd["vol_20"]; scores = dd["scores"]

        all_mean = lv.mean()
        top20 = lv[np.argsort(scores)[-int(len(scores)*0.2):]].mean()
        vol_lo20 = lv[vol20 <= np.percentile(vol20, 20)].mean()
        vol_lo50 = lv[vol20 <= np.percentile(vol20, 50)].mean()
        alpha_hi = lv[scores >= np.percentile(scores, 50)].mean()
        ic = np.corrcoef(scores, lv)[0,1]

        print(f"{date:<12} {csi5:>+6.2f}% {state:>8} {all_mean:>+6.2f}% "
              f"{top20:>+6.2f}% {vol_lo20:>+6.2f}% {vol_lo50:>+6.2f}% {alpha_hi:>+6.2f}% {ic:>+7.4f}")


if __name__ == "__main__":
    main()
