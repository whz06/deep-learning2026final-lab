"""
v3/run_strategies.py — Strategy C, A+C hybrid, and baseline comparison on pre-computed scores.

All strategies share the same score pre-computation.
Parameter sweeps run as post-processing on cached scores.
"""
import os, sys, gc, itertools
import numpy as np
import pandas as pd
import torch

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(SCRIPT_DIR)
V2_DIR = os.path.join(ROOT, "v2")
PARQUET_PATH = os.path.join(ROOT, "processed", "all_data.parquet")
INDEX_PATH   = os.path.join(ROOT, "data", "market", "000300.SH.csv")
CKPT_DIR     = os.path.join(V2_DIR, "checkpoints")

sys.path.insert(0, V2_DIR)
from models.gru import GRURanker

WINDOW_SIZE  = 60
RAW_FEATURES = ["open","high","low","close","vol","amount","pct_chg",
                "turnover_rate","volume_ratio","total_mv"]
TECH_FEATURES = ["macd","macd_signal","rsi","bb_width","bb_pct",
                 "mom_5","mom_20","vol_20"]
GRU_CKPT = "gru_gru_hidden_size=128_num_layers=1_dropout=0.2_lr=0.0003.pt"
TRADE_COST = 0.0013
LIMIT_UP = 9.9
TEST_START, TEST_END = "20260201", "20260531"
SEGMENT_DAYS, STEP = 10, 7

# ── Feature computation ──
def add_tech(df):
    c = df["close"].astype(float)
    e12, e26 = c.ewm(span=12,adjust=False).mean(), c.ewm(span=26,adjust=False).mean()
    df["macd"]=e12-e26; df["macd_signal"]=df["macd"].ewm(span=9,adjust=False).mean()
    d=c.diff(); g=d.clip(lower=0); l=(-d).clip(lower=0)
    rs=g.ewm(alpha=1/14,adjust=False).mean()/(l.ewm(alpha=1/14,adjust=False).mean()+1e-8)
    df["rsi"]=100-100/(1+rs)
    m20,s20=c.rolling(20).mean(),c.rolling(20).std()
    df["bb_width"]=2*s20/(m20+1e-8); df["bb_pct"]=(c-(m20-2*s20))/(4*s20+1e-8)
    df["mom_5"]=c/c.shift(5)-1; df["mom_20"]=c/c.shift(20)-1
    df["vol_20"]=c.pct_change().rolling(20).std()
    return df

def build_day_features(date, stock_series, idx_map):
    feats, codes, vol20_list = [], [], []
    for ts_code, sdf in stock_series.items():
        sd = sdf[sdf["trade_date"] <= date]
        if len(sd) < WINDOW_SIZE+1: continue
        sw = sd.iloc[-WINDOW_SIZE-1:]
        rv = sw[RAW_FEATURES + TECH_FEATURES].values.astype(np.float32)
        if np.isnan(rv).any(): continue
        feats.append(rv[-WINDOW_SIZE:]); codes.append(ts_code)
        vol20_list.append(sw["vol_20"].values[-1])
    if not feats: return None, None, None, None

    fa = np.stack(feats, axis=0); n = len(feats)
    pv=fa[:,-1,6]; av=fa[:,-1,5]; tv=fa[:,-1,7]
    pr=np.argsort(np.argsort(pv)).astype(np.float32)/max(n-1,1)
    ar=np.argsort(np.argsort(av)).astype(np.float32)/max(n-1,1)
    tr=np.argsort(np.argsort(tv)).astype(np.float32)/max(n-1,1)
    ip=idx_map.get(date,0.0); rb=np.full(n,pv-ip,dtype=np.float32)
    cr=np.stack([np.tile(pr[:,None],(1,WINDOW_SIZE)),np.tile(ar[:,None],(1,WINDOW_SIZE)),
                  np.tile(tr[:,None],(1,WINDOW_SIZE)),np.tile(rb[:,None],(1,WINDOW_SIZE))],-1)
    full=np.concatenate([fa,cr],-1)
    m,s=full.mean(axis=0,keepdims=True),full.std(axis=0,keepdims=True)+1e-8
    return (full-m)/s, codes, pv, np.array(vol20_list)


# ── Strategy simulators ──

def sim_baseline(day_data, test_dates, segments, idx_map):
    """n=20, k=2, always rebalance."""
    results = []
    for seg in segments:
        holdings, daily_rets, turnover = [], [], 0
        for pi in range(SEGMENT_DAYS):
            pdate = seg[pi]
            if pdate not in day_data: continue
            dd = day_data[pdate]; scores=dd["scores"]; cv=dd["codes"]
            lv=dd["labels"]; lu=dd["limit_up"]
            if not holdings:
                elig=np.where(~lu)[0]
                if len(elig)>=20: holdings=[cv[i] for i in elig[np.argsort(scores[elig])[-20:]]]
            else:
                hs={c:scores[cv.index(c)] for c in holdings if c in cv}
                if len(hs)>=2:
                    for c in sorted(hs,key=hs.get)[:2]: holdings.remove(c); turnover+=1
                ch=set(holdings)
                cand=[(i,s) for i,(c,s) in enumerate(zip(cv,scores)) if c not in ch and not lu[i]]
                cand.sort(key=lambda x:x[1],reverse=True)
                for i,_ in cand[:2]: holdings.append(cv[i]); turnover+=1
                holdings=holdings[-20:]
            hr=[lv[cv.index(c)] for c in holdings if c in cv]
            if hr:
                dr=float(np.mean(hr))
                if pi>0 and turnover>0: dr-=turnover*TRADE_COST/max(len(holdings),1)
                daily_rets.append(dr)
            turnover=0
        if daily_rets:
            dr=np.array(daily_rets)
            cum=float(np.prod(1+dr/100)-1)*100
            dd_pct=float(np.min(np.cumprod(1+dr/100)/np.maximum.accumulate(np.cumprod(1+dr/100))-1))*100
            results.append(cum)
    return results


def sim_strategy_C(day_data, test_dates, segments, idx_map,
                   bear_thresh=-2.0, vol_pct=0.3, lam_defense=1.0,
                   bear_pos=0.8, bear_lv_ratio=0.5):
    """Three-layer filtering strategy."""
    # Pre-compute CSI300 5-day rolling returns
    csi_dates = sorted(day_data.keys())
    csi_5d = {}
    for i, d in enumerate(csi_dates):
        if i >= 5:
            start = csi_dates[i-4]  # 5 days ago (inclusive)
            rets = [idx_map.get(csi_dates[j], 0) for j in range(i-4, i+1)]
            csi_5d[d] = float(np.prod(1 + np.array(rets)/100) - 1) * 100
        else:
            csi_5d[d] = 0.0

    results = []
    for seg in segments:
        holdings, daily_rets, turnover = [], [], 0
        for pi in range(SEGMENT_DAYS):
            pdate = seg[pi]
            if pdate not in day_data: continue
            dd = day_data[pdate]; scores=dd["scores"]; cv=dd["codes"]
            lv=dd["labels"]; lu=dd["limit_up"]; vol20=dd["vol_20"]

            # Layer 1: Market state
            csi5 = csi_5d.get(pdate, 0.0)
            if csi5 > 1.0:    state = "bull"
            elif csi5 >= bear_thresh: state = "neutral"
            else:              state = "bear"

            # Layer 2: Stock pool
            z_alpha = (scores - scores.mean()) / (scores.std() + 1e-8)
            z_vol = (vol20 - vol20.mean()) / (vol20.std() + 1e-8)
            vol_lo = np.percentile(vol20, vol_pct * 100)
            alpha_hi = np.percentile(scores, 50)

            if state == "bear":
                pool_mask = (vol20 <= vol_lo) & (scores >= alpha_hi) & (~lu)
            else:
                pool_mask = ~lu

            # Layer 3: Score adjustment
            if state == "bull":
                adj_score = z_alpha
            elif state == "neutral":
                adj_score = z_alpha - 0.2 * z_vol
            else:  # bear
                adj_score = z_alpha - lam_defense * z_vol

            # Apply pool mask
            valid_idx = np.where(pool_mask)[0]
            if len(valid_idx) == 0:
                valid_idx = np.where(~lu)[0]  # fallback

            adj_valid = adj_score[valid_idx]
            cv_valid = [cv[i] for i in valid_idx]

            # Position sizing
            if state == "bear":
                n_lv = max(1, int(bear_pos * bear_lv_ratio * 20))
                n_alpha = max(1, int(bear_pos * (1 - bear_lv_ratio) * 20))
                total_n = n_lv + n_alpha
            else:
                total_n = 20

            # Execute trades
            top_idx = np.argsort(adj_valid)[-total_n:]
            new_holdings = [cv_valid[i] for i in top_idx]

            if state == "bear":
                # Compute turnover from holdings change
                old_set = set(holdings)
                new_set = set(new_holdings)
                sold = old_set - new_set
                bought = new_set - old_set
                turnover = max(len(sold), len(bought))
            else:
                old_set = set(holdings)
                new_set = set(new_holdings)
                sold = old_set - new_set
                bought = new_set - old_set
                turnover = max(len(sold), len(bought))

            holdings = new_holdings

            hr = [lv[cv.index(c)] for c in holdings if c in cv]
            if hr:
                dr = float(np.mean(hr))
                if pi > 0 and turnover > 0:
                    dr -= turnover * TRADE_COST / max(len(holdings), 1)
                daily_rets.append(dr)

        if daily_rets:
            dr = np.array(daily_rets)
            cum = float(np.prod(1 + dr/100) - 1) * 100
            results.append(cum)
    return results


def sim_strategy_AC(day_data, test_dates, segments, idx_map,
                    lam_normal=0.3, lam_defense=1.0, alpha_thresh=-0.2,
                    bear_pos=0.8):
    """A+C hybrid: continuous vol penalty + position management."""
    results = []
    for seg in segments:
        holdings, daily_rets, turnover = [], [], 0
        for pi in range(SEGMENT_DAYS):
            pdate = seg[pi]
            if pdate not in day_data: continue
            dd = day_data[pdate]; scores=dd["scores"]; cv=dd["codes"]
            lv=dd["labels"]; lu=dd["limit_up"]; vol20=dd["vol_20"]

            z_alpha = (scores - scores.mean()) / (scores.std() + 1e-8)
            z_vol = (vol20 - vol20.mean()) / (vol20.std() + 1e-8)

            # Detect defense mode
            # z_alpha of top-20 stocks
            elig = np.where(~lu)[0]
            if len(elig) >= 20:
                top20_idx = elig[np.argsort(z_alpha[elig])[-20:]]
                top20_alpha_mean = z_alpha[top20_idx].mean()
            else:
                top20_alpha_mean = 0

            defense = top20_alpha_mean < alpha_thresh
            lam = lam_defense if defense else lam_normal
            adj_score = z_alpha - lam * z_vol

            # Position
            if defense:
                n_hold = int(bear_pos * 20)
            else:
                n_hold = 20

            elig = np.where(~lu)[0]
            if len(elig) >= n_hold:
                top_idx = elig[np.argsort(adj_score[elig])[-n_hold:]]
                new_holdings = [cv[i] for i in top_idx]
            else:
                new_holdings = [cv[i] for i in elig]

            old_set = set(holdings)
            new_set = set(new_holdings)
            turnover = max(len(old_set - new_set), len(new_set - old_set))
            holdings = new_holdings

            hr = [lv[cv.index(c)] for c in holdings if c in cv]
            if hr:
                dr = float(np.mean(hr))
                if pi > 0 and turnover > 0:
                    dr -= turnover * TRADE_COST / max(len(holdings), 1)
                daily_rets.append(dr)

        if daily_rets:
            dr = np.array(daily_rets)
            cum = float(np.prod(1 + dr/100) - 1) * 100
            results.append(cum)
    return results


# ── Main ──

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[device] {device}")

    # Load model
    m = GRURanker(22,128,1,0.2)
    m.load_state_dict(torch.load(os.path.join(CKPT_DIR, GRU_CKPT), map_location=device, weights_only=True))
    m.to(device).eval()

    # Load data
    print("[data] Loading ...")
    df = pd.read_parquet(PARQUET_PATH); df["trade_date"] = df["trade_date"].astype(str)
    dates_all = sorted(df["trade_date"].unique())
    tc = [d for d in dates_all if d >= TEST_START]
    lb = max(0, dates_all.index(tc[0]) - WINDOW_SIZE - 1 - 20)
    df = df[df["trade_date"] >= dates_all[lb]]
    idx_df = pd.read_csv(INDEX_PATH, dtype={"trade_date": str})
    idx_map = dict(zip(idx_df["trade_date"], idx_df["pct_chg"].astype(float)))

    print("[feat] Computing tech indicators ...")
    stocks = sorted(df["ts_code"].unique())
    stock_series = {}
    for i, ts in enumerate(stocks):
        sdf = df[df["ts_code"]==ts].sort_values("trade_date").reset_index(drop=True)
        stock_series[ts] = add_tech(sdf)
        if (i+1)%1000==0: print(f"  ... {i+1}/{len(stocks)}")

    test_dates = [d for d in dates_all if TEST_START <= d <= TEST_END]
    segments = []
    i = 0
    while i + SEGMENT_DAYS + 1 <= len(test_dates):
        segments.append(test_dates[i:i+SEGMENT_DAYS+1])
        i += STEP
    print(f"[dates] {len(test_dates)} days, {len(segments)} segments")

    # Pre-compute scores + vol20
    print("[pred] Computing predictions for all test dates ...")
    day_data = {}
    for di, date in enumerate(test_dates):
        if di == len(test_dates)-1: break
        next_date = test_dates[di+1]
        features, codes, pct_now, vol20_arr = build_day_features(date, stock_series, idx_map)
        if features is None: continue
        scores = m(torch.from_numpy(features).to(device)).detach().cpu().numpy()
        torch.cuda.empty_cache()

        vl, vi = [], []
        for ci, c in enumerate(codes):
            r = stock_series[c][stock_series[c]["trade_date"]==next_date]
            if len(r)>0: vl.append(r["pct_chg"].values[0]); vi.append(ci)
        if not vi: continue

        day_data[date] = {"scores": scores[vi], "codes": [codes[i] for i in vi],
                          "labels": np.array(vl), "limit_up": features[vi,-1,6]>=LIMIT_UP,
                          "vol_20": vol20_arr[vi]}
        if (di+1)%10==0: print(f"  ... {di+1}/{len(test_dates)-1}")
    print(f"[pred] {len(day_data)} dates computed\n")
    del m; torch.cuda.empty_cache()

    # CSI300 per segment
    csi_seg = []
    for seg in segments:
        csi_daily=[idx_map.get(seg[j+1],0.0) for j in range(SEGMENT_DAYS)]
        csi_seg.append(float(np.prod(1+np.array(csi_daily)/100)-1)*100)

    # ── Baseline ──
    print("=== BASELINE (n=20,k=2) ===")
    b_ret = sim_baseline(day_data, test_dates, segments, idx_map)
    b_arr = np.array(b_ret)
    b_win = (b_arr > 0).sum()
    exc_bl = b_arr - np.array(csi_seg)
    print(f"Mean: {b_arr.mean():+.2f}% | CSI300: {np.mean(csi_seg):.2f}% | "
          f"Excess: {exc_bl.mean():+.2f}% | Std: {b_arr.std():.2f}% | W/L: {b_win}/{len(b_arr)}\n")

    # ── Strategy C sweep ──
    print("=== STRATEGY C SWEEP ===")
    c_best = {"bear_thresh": 0, "vol_pct": 0, "lam_defense": 0, "bear_pos": 0,
              "bear_lv_ratio": 0, "mean": -999, "excess": -999}
    c_results = []

    for bt in [-1.5, -2.0, -2.5]:
        for vp in [0.2, 0.3]:
            for ld in [0.5, 1.0, 1.5]:
                for bp in [0.75, 0.80, 0.85]:
                    for blr in [0.4, 0.5, 0.6]:
                        ret = sim_strategy_C(day_data, test_dates, segments, idx_map,
                                            bt, vp, ld, bp, blr)
                        if len(ret)==0: continue
                        arr = np.array(ret)
                        exc = arr - np.array(csi_seg)
                        entry = {"bt":bt,"vp":vp,"ld":ld,"bp":bp,"blr":blr,
                                 "mean":arr.mean(), "exc":exc.mean(), "std":arr.std(),
                                 "win":(arr>0).sum()}
                        c_results.append(entry)
                        if exc.mean() > c_best["excess"]:
                            c_best = {**entry, "excess": exc.mean()}

    c_df = pd.DataFrame(c_results)
    print(f"Tested {len(c_results)} configs")
    top5_c = c_df.nlargest(5, "exc")
    print(top5_c.to_string(index=False))
    print(f"\nC Best: bt={c_best['bt']}, vp={c_best['vp']}, ld={c_best['ld']}, "
          f"bp={c_best['bp']}, blr={c_best['blr']}")
    print(f"  Mean={c_best['mean']:+.2f}%, Excess={c_best['excess']:+.2f}%, "
          f"Std={c_best['std']:.2f}%, W/L={c_best['win']}/{len(ret)}\n")

    # ── Strategy A+C sweep ──
    print("=== STRATEGY A+C SWEEP ===")
    ac_best = {"lam_normal": 0, "lam_defense": 0, "alpha_thresh": 0, "bear_pos": 0,
               "mean": -999, "excess": -999}
    ac_results = []

    for ln in [0, 0.3, 0.5]:
        for ld in [0.5, 1.0, 1.5]:
            for at in [-0.3, -0.2, -0.1]:
                for bp in [0.75, 0.80, 0.85]:
                    ret = sim_strategy_AC(day_data, test_dates, segments, idx_map, ln, ld, at, bp)
                    if len(ret)==0: continue
                    arr = np.array(ret)
                    exc = arr - np.array(csi_seg)
                    entry = {"ln":ln,"ld":ld,"at":at,"bp":bp,
                             "mean":arr.mean(),"exc":exc.mean(),"std":arr.std(),
                             "win":(arr>0).sum()}
                    ac_results.append(entry)
                    if exc.mean() > ac_best["excess"]:
                        ac_best = {**entry, "excess": exc.mean()}

    ac_df = pd.DataFrame(ac_results)
    print(f"Tested {len(ac_results)} configs")
    top5_ac = ac_df.nlargest(5, "exc")
    print(top5_ac.to_string(index=False))
    print(f"\nA+C Best: ln={ac_best['ln']}, ld={ac_best['ld']}, at={ac_best['at']}, bp={ac_best['bp']}")
    print(f"  Mean={ac_best['mean']:+.2f}%, Excess={ac_best['excess']:+.2f}%, "
          f"Std={ac_best['std']:.2f}%, W/L={ac_best['win']}/{len(ret)}\n")

    # ── Final Comparison ──
    print("\n" + "="*80)
    print("  FINAL COMPARISON")
    print("="*80)
    print(f"{'Strategy':<16} {'Mean':>7} {'Excess':>7} {'Std':>6} {'W/L':>5}")
    print("-"*50)
    print(f"{'baseline(n=20)':<16} {b_arr.mean():>+6.2f}% {exc_bl.mean():>+6.2f}% "
          f"{b_arr.std():>5.2f}% {b_win}/{len(b_arr)}")

    # Best C
    best_c_ret = np.array([r["mean"] for r in c_results if
                           r["bt"]==c_best["bt"] and r["vp"]==c_best["vp"] and
                           r["ld"]==c_best["ld"] and r["bp"]==c_best["bp"] and
                           r["blr"]==c_best["blr"]])
    if len(best_c_ret)>0:
        print(f"{'Strategy C':<16} {best_c_ret.mean():>+6.2f}% {c_best['excess']:>+6.2f}% "
              f"{best_c_ret.std():>5.2f}% {c_best['win']}/{len(best_c_ret)}")

    # Best A+C
    best_ac_ret = np.array([r["mean"] for r in ac_results if
                            r["ln"]==ac_best["ln"] and r["ld"]==ac_best["ld"] and
                            r["at"]==ac_best["at"] and r["bp"]==ac_best["bp"]])
    if len(best_ac_ret)>0:
        print(f"{'Strategy A+C':<16} {best_ac_ret.mean():>+6.2f}% {ac_best['excess']:>+6.2f}% "
              f"{best_ac_ret.std():>5.2f}% {ac_best['win']}/{len(best_ac_ret)}")
    print("="*80)

    # Save best configs
    import json
    with open(os.path.join(SCRIPT_DIR, "best_strategies.json"), "w") as f:
        json.dump({"baseline": {"mean": float(b_arr.mean()), "excess": float(exc_bl.mean()),
                                "std": float(b_arr.std()), "win": int(b_win)},
                    "C": c_best, "AC": ac_best}, f, indent=2)
    print("\nSaved best_strategies.json")


if __name__ == "__main__":
    main()
