"""
v2/testing/walkforward_v2.py — Multi-strategy walk-forward with detailed per-day logs.

Strategies tested on same data:
  1. baseline:    n=20, k=2, always rebalance
  2. tight:       n=8,  k=2, always rebalance
  3. market_aw:   n=8,  k=2, skip rebalance if market up >0.5%
  4. threshold:   n=8,  k=2, only buy z_score > 0 (above median)
  5. full:        n=8,  k=2, market_aw + threshold

Outputs: comparison table + per-strategy daily detail.
"""
import os, sys, gc, argparse
import numpy as np, pandas as pd, torch

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
V2_DIR = os.path.dirname(SCRIPT_DIR)
ROOT = os.path.dirname(V2_DIR)
PARQUET_PATH = os.path.join(ROOT, "processed", "all_data.parquet")
INDEX_PATH   = os.path.join(ROOT, "data", "market", "000300.SH.csv")
CKPT_DIR = os.path.join(V2_DIR, "checkpoints")

sys.path.insert(0, V2_DIR)
from models.gru import GRURanker

WINDOW_SIZE  = 60
RAW_FEATURES = ["open","high","low","close","vol","amount","pct_chg",
                "turnover_rate","volume_ratio","total_mv"]
TECH_FEATURES = ["macd","macd_signal","rsi","bb_width","bb_pct",
                 "mom_5","mom_20","vol_20"]
GRU_CKPT = "gru_gru_hidden_size=128_num_layers=1_dropout=0.2_lr=0.0003.pt"

TRADE_COST = 0.0013
LIMIT_UP   = 9.9
TEST_START, TEST_END = "20260201", "20260531"
SEGMENT_DAYS, STEP = 10, 7


# ── Tech indicators ──
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


# ── Feature building ──
def build_day_features(date, stock_series, idx_map):
    feats, codes = [], []
    for ts_code, sdf in stock_series.items():
        sd = sdf[sdf["trade_date"] <= date]
        if len(sd) < WINDOW_SIZE+1: continue
        sw = sd.iloc[-WINDOW_SIZE-1:]
        rv = sw[RAW_FEATURES + TECH_FEATURES].values.astype(np.float32)
        if np.isnan(rv).any(): continue
        feats.append(rv[-WINDOW_SIZE:]); codes.append(ts_code)
    if not feats: return None, None, None

    fa = np.stack(feats, axis=0); n = len(feats)
    pv = fa[:,-1,6]; av = fa[:,-1,5]; tv = fa[:,-1,7]
    pr = np.argsort(np.argsort(pv)).astype(np.float32)/max(n-1,1)
    ar = np.argsort(np.argsort(av)).astype(np.float32)/max(n-1,1)
    tr = np.argsort(np.argsort(tv)).astype(np.float32)/max(n-1,1)
    ip = idx_map.get(date,0.0); rb = np.full(n, pv - ip, dtype=np.float32)
    cr = np.stack([np.tile(pr[:,None],(1,WINDOW_SIZE)),
                   np.tile(ar[:,None],(1,WINDOW_SIZE)),
                   np.tile(tr[:,None],(1,WINDOW_SIZE)),
                   np.tile(rb[:,None],(1,WINDOW_SIZE))], -1)
    full = np.concatenate([fa, cr], -1)
    m,s = full.mean(axis=0,keepdims=True), full.std(axis=0,keepdims=True)+1e-8
    full = (full-m)/s
    return full, codes, pv  # pv = last-day pct_chg for limit-up filter


# ── Strategy simulators ──
# Each returns (holdings after sim, daily_returns, turnover_count)

def strategy_baseline(scores, cv, lv, limit_up, holdings, pi, n=20, k=2):
    """n=20, k=2, always rebalance."""
    turnover = 0
    if not holdings:
        elig = np.where(~limit_up)[0]
        if len(elig) >= n:
            idx = elig[np.argsort(scores[elig])[-n:]]
            holdings = [cv[i] for i in idx]
    else:
        hs = {c: scores[cv.index(c)] for c in holdings if c in cv}
        if len(hs) >= k:
            for c in sorted(hs, key=hs.get)[:k]: holdings.remove(c); turnover += 1
        ch = set(holdings)
        cand = [(i,s) for i,(c,s) in enumerate(zip(cv,scores)) if c not in ch and not limit_up[i]]
        cand.sort(key=lambda x:x[1],reverse=True)
        for i,_ in cand[:k]: holdings.append(cv[i]); turnover += 1
        holdings = holdings[-n:]
    return holdings, turnover


def strategy_tight(scores, cv, lv, limit_up, holdings, pi, n=8, k=2):
    """n=8, k=2, always rebalance."""
    turnover = 0
    if not holdings:
        elig = np.where(~limit_up)[0]
        if len(elig) >= n:
            idx = elig[np.argsort(scores[elig])[-n:]]
            holdings = [cv[i] for i in idx]
    else:
        hs = {c: scores[cv.index(c)] for c in holdings if c in cv}
        if len(hs) >= k:
            for c in sorted(hs, key=hs.get)[:k]: holdings.remove(c); turnover += 1
        ch = set(holdings)
        cand = [(i,s) for i,(c,s) in enumerate(zip(cv,scores)) if c not in ch and not limit_up[i]]
        cand.sort(key=lambda x:x[1],reverse=True)
        for i,_ in cand[:k]: holdings.append(cv[i]); turnover += 1
        holdings = holdings[-n:]
    return holdings, turnover


def strategy_market_aw(scores, cv, lv, limit_up, holdings, pi, n=8, k=2):
    """n=8, k=2, skip rebalance if all-stock mean return > 0.5% (bull day)."""
    all_mean = lv[~limit_up].mean() if (~limit_up).sum() > 0 else 0
    if all_mean > 0.5 and holdings:
        return holdings, 0  # no rebalance, no turnover cost

    turnover = 0
    if not holdings:
        elig = np.where(~limit_up)[0]
        if len(elig) >= n:
            idx = elig[np.argsort(scores[elig])[-n:]]
            holdings = [cv[i] for i in idx]
    else:
        hs = {c: scores[cv.index(c)] for c in holdings if c in cv}
        if len(hs) >= k:
            for c in sorted(hs, key=hs.get)[:k]: holdings.remove(c); turnover += 1
        ch = set(holdings)
        cand = [(i,s) for i,(c,s) in enumerate(zip(cv,scores)) if c not in ch and not limit_up[i]]
        cand.sort(key=lambda x:x[1],reverse=True)
        for i,_ in cand[:k]: holdings.append(cv[i]); turnover += 1
        holdings = holdings[-n:]
    return holdings, turnover


def strategy_threshold(scores, cv, lv, limit_up, holdings, pi, n=8, k=2):
    """n=8, k=2, only buy stocks with Z-score > 0 (above daily median)."""
    z = (scores - scores.mean()) / (scores.std() + 1e-8)
    turnover = 0
    if not holdings:
        elig = np.where((~limit_up) & (z > 0))[0]
        if len(elig) >= n:
            idx = elig[np.argsort(scores[elig])[-n:]]
            holdings = [cv[i] for i in idx]
        elif len(elig) > 0:
            # Fall through to threshold range
            idx = np.argsort(scores[elig])[-len(elig):]
            holdings = [cv[i] for i in idx]
    else:
        hs = {c: scores[cv.index(c)] for c in holdings if c in cv}
        if len(hs) >= k:
            for c in sorted(hs, key=hs.get)[:k]: holdings.remove(c); turnover += 1
        ch = set(holdings)
        cand = [(i,s) for i,(c,s) in enumerate(zip(cv,scores)) if c not in ch and not limit_up[i] and z[i] > 0]
        cand.sort(key=lambda x:x[1],reverse=True)
        for i,_ in cand[:k]: holdings.append(cv[i]); turnover += 1
        holdings = holdings[-n:]
    return holdings, turnover


def strategy_full(scores, cv, lv, limit_up, holdings, pi, n=8, k=2):
    """market_aw + threshold combined."""
    all_mean = lv[~limit_up].mean() if (~limit_up).sum() > 0 else 0
    if all_mean > 0.5 and holdings:
        return holdings, 0

    z = (scores - scores.mean()) / (scores.std() + 1e-8)
    turnover = 0
    if not holdings:
        elig = np.where((~limit_up) & (z > 0))[0]
        if len(elig) >= n:
            idx = elig[np.argsort(scores[elig])[-n:]]
            holdings = [cv[i] for i in idx]
        elif len(elig) > 0:
            idx = np.argsort(scores[elig])[-min(len(elig), n):]
            holdings = [cv[i] for i in idx]
    else:
        hs = {c: scores[cv.index(c)] for c in holdings if c in cv}
        if len(hs) >= k:
            for c in sorted(hs, key=hs.get)[:k]: holdings.remove(c); turnover += 1
        ch = set(holdings)
        cand = [(i,s) for i,(c,s) in enumerate(zip(cv,scores)) if c not in ch and not limit_up[i] and z[i] > 0]
        cand.sort(key=lambda x:x[1],reverse=True)
        for i,_ in cand[:k]: holdings.append(cv[i]); turnover += 1
        holdings = holdings[-n:]
    return holdings, turnover


STRATEGIES = {
    "baseline":   (strategy_baseline, 20),
    "tight":      (strategy_tight, 8),
    "market_aw":  (strategy_market_aw, 8),
    "threshold":  (strategy_threshold, 8),
    "full":       (strategy_full, 8),
}


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
    print(f"  {len(df)} rows, {df['trade_date'].nunique()} dates, {df['ts_code'].nunique()} stocks")

    # CSI300
    idx_df = pd.read_csv(INDEX_PATH, dtype={"trade_date": str})
    idx_map = dict(zip(idx_df["trade_date"], idx_df["pct_chg"].astype(float)))

    # Tech indicators
    print("[feat] Computing tech indicators ...")
    stocks = sorted(df["ts_code"].unique())
    stock_series = {}
    for i, ts in enumerate(stocks):
        stock_series[ts] = add_tech(df[df["ts_code"]==ts].sort_values("trade_date").reset_index(drop=True))
        if (i+1) % 1000 == 0: print(f"  ... {i+1}/{len(stocks)}")
    print(f"  {len(stocks)} stocks done")

    # Test dates & segments
    test_dates = [d for d in dates_all if TEST_START <= d <= TEST_END]
    print(f"[dates] {len(test_dates)} days ({test_dates[0]}~{test_dates[-1]})")

    segments = []
    i = 0
    while i + SEGMENT_DAYS + 1 <= len(test_dates):
        segments.append(test_dates[i:i + SEGMENT_DAYS + 1])
        i += STEP
    print(f"[segments] {len(segments)} segments\n")

    # ── Pre-compute ALL scores for ALL test dates ──
    print("[pred] Computing predictions for all test dates ...")
    day_data = {}   # date -> {scores, codes, labels, next_date, pct_today, limit_up}
    for di, date in enumerate(test_dates):
        if di == len(test_dates)-1: break  # last date has no next day
        next_date = test_dates[di+1]
        features, codes, pct_now = build_day_features(date, stock_series, idx_map)
        if features is None: continue

        scores = m(torch.from_numpy(features).to(device)).detach().cpu().numpy()
        torch.cuda.empty_cache()

        # Actual next-day returns
        vl, vi = [], []
        for ci, c in enumerate(codes):
            r = stock_series[c][stock_series[c]["trade_date"]==next_date]
            if len(r)>0: vl.append(r["pct_chg"].values[0]); vi.append(ci)
        if not vi: continue

        day_data[date] = {
            "scores": scores[vi],
            "codes": [codes[i] for i in vi],
            "labels": np.array(vl),
            "next_date": next_date,
            "pct_today": features[vi, -1, 6],
            "limit_up": features[vi, -1, 6] >= LIMIT_UP,
        }
        if (di+1) % 10 == 0: print(f"  ... {di+1}/{len(test_dates)-1}")
    print(f"[pred] {len(day_data)} dates computed\n")

    del m; torch.cuda.empty_cache()

    # ── Run each strategy on same data ──
    all_summaries = {}

    for sname, (strat_fn, n_hold) in STRATEGIES.items():
        print(f"\n{'='*60}\n  Strategy: {sname} (n={n_hold})\n{'='*60}")
        seg_results = []
        daily_log = []

        for seg_idx, seg in enumerate(segments):
            holdings = []
            total_turnover = 0
            daily_rets = []

            for pi in range(SEGMENT_DAYS):
                pdate = seg[pi]
                if pdate not in day_data: continue
                dd = day_data[pdate]

                holdings, to = strat_fn(dd["scores"], dd["codes"], dd["labels"],
                                        dd["limit_up"], holdings, pi, n_hold)
                total_turnover += to

                hr = [dd["labels"][dd["codes"].index(c)] for c in holdings if c in dd["codes"]]
                if hr:
                    dr = float(np.mean(hr))
                    if pi > 0 and to > 0:
                        dr -= to * TRADE_COST / max(len(holdings), 1)
                    daily_rets.append(dr)

                daily_log.append({
                    "seg": seg_idx+1, "day": pi+1, "date": pdate,
                    "held_n": len(holdings), "turnover": to,
                    "daily_ret": dr if hr else 0,
                })

            if daily_rets:
                dr = np.array(daily_rets)
                cum = float(np.prod(1 + dr / 100) - 1) * 100
                dd_pct = float(np.min(np.cumprod(1 + dr / 100) /
                              np.maximum.accumulate(np.cumprod(1 + dr / 100)) - 1)) * 100
                seg_results.append({
                    "seg": seg_idx+1, "start": seg[0], "end": seg[-1],
                    "cum_ret": cum, "max_dd": dd_pct,
                    "turnover": total_turnover, "n_days": len(daily_rets),
                })

        # Segment-level summary
        if not seg_results: continue
        rets = np.array([s["cum_ret"] for s in seg_results])
        dd_arr = np.array([s["max_dd"] for s in seg_results])
        t_avg = np.mean([s["turnover"] for s in seg_results]) / SEGMENT_DAYS

        # CSI300 per segment
        csi_rets = []
        for seg in segments:
            csi_daily = [idx_map.get(seg[j+1], 0.0) for j in range(SEGMENT_DAYS)]
            csi_rets.append(float(np.prod(1 + np.array(csi_daily) / 100) - 1) * 100)
        csi_rets = np.array(csi_rets)

        excess = rets - csi_rets
        wins = (rets > 0).sum()

        all_summaries[sname] = {
            "mean": rets.mean(), "std": rets.std(),
            "max": rets.max(), "min": rets.min(),
            "csi_mean": csi_rets.mean(),
            "excess_mean": excess.mean(),
            "wins": wins, "total": len(rets),
            "dd_mean": dd_arr.mean(),
            "dd_max": dd_arr.min() if len(dd_arr)>0 else 0,
            "avg_turnover": t_avg,
            "avg_held": np.mean([l["held_n"] for l in daily_log]) if daily_log else 0,
            "seg_results": seg_results,
            "daily_log": daily_log,
        }

        # Print per-segment
        for s in seg_results:
            csi_c = csi_rets[s["seg"]-1]
            ex = s["cum_ret"] - csi_c
            w = "W" if s["cum_ret"] > 0 else "L"
            print(f"  Seg {s['seg']:2d} | {s['start']}→{s['end']} | "
                  f"Ret:{s['cum_ret']:+6.2f}% | CSI:{csi_c:+6.2f}% | "
                  f"Ex:{ex:+6.2f}% | DD:{s['max_dd']:+5.2f}% | "
                  f"TO:{s['turnover']}/10d | {w}")

    # ── Final Comparison ──
    print(f"\n\n{'='*80}")
    print(f"  STRATEGY COMPARISON — 10-segment × 10-day on 2026 unseen data")
    print(f"{'='*80}")
    print(f"{'Strategy':<12} {'Mean':>7} {'Std':>6} {'Max':>7} {'Min':>7} "
          f"{'CSI300':>7} {'Excess':>7} {'DD_avg':>6} {'W/L':>5} {'TO/d':>5} {'Held':>5}")
    print(f"{'-'*80}")
    for name, r in all_summaries.items():
        print(f"{name:<12} {r['mean']:>+6.2f}% {r['std']:>5.2f}% "
              f"{r['max']:>+6.2f}% {r['min']:>+6.2f}% "
              f"{r['csi_mean']:>+6.2f}% {r['excess_mean']:>+6.2f}% "
              f"{r['dd_mean']:>+5.2f}% {r['wins']}/{r['total']:<3} "
              f"{r['avg_turnover']:>4.1f} {r['avg_held']:>4.0f}")
    print(f"{'='*80}")

    # ── Detailed daily log for best strategy ──
    if all_summaries:
        best_name = max(all_summaries, key=lambda n: all_summaries[n]["excess_mean"])
        best = all_summaries[best_name]
        print(f"\n\n[detail] Best strategy: {best_name} — daily breakdown (first segment)")
        log = best["daily_log"]
        seg1 = [l for l in log if l["seg"] == 1]
        print(f"{'Day':>4} {'Date':<12} {'Held':>5} {'TO':>4} {'Ret':>8}")
        print(f"{'-'*35}")
        for l in seg1:
            print(f"{l['day']:>4} {l['date']:<12} {l['held_n']:>5} {l['turnover']:>4} {l['daily_ret']:>+7.2f}%")

    gc.collect(); torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
