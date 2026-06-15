"""plot_competition_real.py — 实际成交+模型轮动重建竞赛收益

6月1日: 实际买入10只股票 → 等权日收益
6月2日: 6/1收盘模型Top5 → 全换仓 → 等权日收益  
6月3-12日: N=5 K=3轮动 (前日信号→次日调仓→当日收益)
"""
import os, sys, numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(SCRIPT_DIR)
OUTDIR = os.path.join(SCRIPT_DIR, "figures")
os.makedirs(OUTDIR, exist_ok=True)

PARQUET = os.path.join(ROOT, "v7", "results", "daily_scores_spatial_t1.parquet")
PARQUET_ALL = os.path.join(ROOT, "processed", "all_data.parquet")

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["SimHei", "DejaVu Sans", "Arial"],
    "axes.unicode_minus": False,
    "figure.dpi": 150, "savefig.dpi": 150,
    "savefig.bbox": "tight", "figure.facecolor": "white",
})
C = {"pos": "#4CAF50", "neg": "#F44336", "v7": "#EF5350", "csi300": "#9E9E9E"}

JUNE1 = ["003030.SZ","603029.SH","300927.SZ","603686.SH","688616.SH",
         "603048.SH","002877.SZ","688151.SH","600319.SH","605033.SH"]

CSI300 = {"20260601":-0.98,"20260602":1.45,"20260603":0.49,"20260604":-0.69,
          "20260605":-1.79,"20260608":-2.14,"20260609":1.87,"20260610":-1.11,
          "20260611":-0.55,"20260612":1.16}

# CSI5d from decision.log (5日滚动, 用于Strategy B风控)
CSI5D = {"20260529":0.98, "20260601":-1.57, "20260602":-1.57, "20260603":0.64,
         "20260604":-0.17, "20260605":-1.51, "20260608":0, "20260609":0,
         "20260610":0, "20260611":0}


def main():
    scores = pd.read_parquet(PARQUET)
    scores["trade_date"] = scores["trade_date"].astype(str)
    ret_all = pd.read_parquet(PARQUET_ALL)
    ret_all["trade_date"] = ret_all["trade_date"].astype(str)
    rlk = {d: g.set_index("ts_code")["pct_chg"]/100.0 for d,g in ret_all.groupby("trade_date")}
    ad = sorted(ret_all["trade_date"].unique())

    def nd(d):
        try: return ad[ad.index(d)+1]
        except: return None

    signals = ["20260601","20260602","20260603","20260604","20260605",
               "20260608","20260609","20260610","20260611"]
    held = set()
    rows = []

    for i, sd in enumerate(signals):
        rd = nd(sd)
        if rd is None: break
        sc = scores[scores["trade_date"]==sd].set_index("ts_code")["score"]
        dr = rlk.get(rd, pd.Series(dtype=float))

        if i == 0:
            eff = set(JUNE1) & set(dr.index)
            pr = dr[list(eff)].mean() if eff else 0.0
            cost = (0.00076+0.00026)*10/10
            held = eff
        elif i == 1:
            N = 5
            top = set(sc.nlargest(N).index) & set(dr.index)
            pr = dr[list(top)].mean() if top else 0.0
            cost = (0.00076+0.00026)*N/N
            held = top
        else:
            N, K = 5, 3
            csi5d = CSI5D.get(sd, 0)
            if csi5d < -1.0:
                N = max(1, int(N * 0.8))  # Strategy B: 80%仓位
            hv = held & set(dr.index)
            pr = dr[list(hv)].mean() if hv else 0.0
            top = set(sc.nlargest(N).index) & set(dr.index)
            rk = sorted(hv, key=lambda x: sc.get(x,-1e9))
            ts = set(rk[:K]) & hv
            tb = set(sorted(top-hv, key=lambda x: sc.get(x,-1e9), reverse=True)[:K])
            cost = (0.00076+0.00026)*max(len(ts),len(tb))/N
            held = (hv-ts)|tb

        rows.append({"return_date":rd, "port_ret":pr-cost})
        print(f"  {sd}->{rd}  N={len(held)}  ret={(pr-cost)*100:+.2f}%")

    df = pd.DataFrame(rows)
    df["cum"] = (1+df["port_ret"]).cumprod()
    cr = [CSI300.get(d,0.0)/100.0 for d in df["return_date"]]
    cc = (1+np.array(cr)).cumprod()

    fig, (ax1, ax2) = plt.subplots(2,1,figsize=(13,7.5),sharex=True,
                                    gridspec_kw={"height_ratios":[1.1,1.3]})
    fig.subplots_adjust(top=0.92)
    x = range(len(df))
    dp = df["port_ret"].values*100
    bc = [C["pos"] if r>0 else C["neg"] for r in dp]

    ax1.bar(x, dp, color=bc, edgecolor="white", width=0.55, zorder=3)
    ax1.axhline(y=0, color="black", linewidth=0.5)
    for i,(d,r) in enumerate(zip(df["return_date"],dp)):
        ax1.text(i, r+(0.4 if r>=0 else -0.8), f"{r:+.1f}%",
                 ha="center", fontsize=9, fontweight="bold",
                 color=C["pos"] if r>=0 else C["neg"])
    ax1.set_ylabel("日收益率 (%)", fontsize=12)
    ax1.set_title("模拟交易竞赛期 — 每日收益 (2026年6月1-12日)", fontsize=14, fontweight="bold")
    ax1.grid(axis="y", alpha=0.2)

    ax2.plot(x, df["cum"].values, "o-", color=C["v7"], linewidth=2.5, markersize=8,
             label="V7 Spatial 策略", zorder=3)
    ax2.plot(x, cc, "s--", color=C["csi300"], linewidth=2, markersize=6,
             label="沪深300基准", zorder=2)
    ax2.axhline(y=1.0, color="black", linewidth=0.5, linestyle="--", alpha=0.3)
    for i,(d,c) in enumerate(zip(df["return_date"],df["cum"])):
        ax2.annotate(f"{c:.4f}", (i,c), textcoords="offset points",
                     xytext=(0,12), fontsize=8, ha="center", fontweight="bold")
    ax2.set_xticks(x)
    ax2.set_xticklabels([f"6/{d[6:8]}" for d in df["return_date"]], fontsize=10)
    ax2.set_ylabel("累计收益 (倍)", fontsize=12)
    ax2.set_xlabel("日期", fontsize=12)
    ax2.grid(alpha=0.2)
    ax2.legend(fontsize=10, loc="upper left")

    p = os.path.join(OUTDIR, "fig_competition.png")
    fig.savefig(p, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[*] {p}")
    print(f"[*] 策略: {(df['cum'].iloc[-1]-1)*100:+.2f}%  沪深300: {(cc[-1]-1)*100:+.2f}%")


if __name__ == "__main__":
    main()
