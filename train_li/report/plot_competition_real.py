"""plot_competition_real.py — 多策略竞赛收益对比

上层: 三策略日收益率折线图
  实际执行: 6/1=10只实盘 → 6/2=Top5 → 6/3起 N5K3+SB
  N=5 K=3: 全程 N5K3+SB
  N=10 K=3: 全程 N10K3+SB
下层: 实际策略累计 vs 沪深300
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

JUNE1 = ["003030.SZ","603029.SH","300927.SZ","603686.SH","688616.SH",
         "603048.SH","002877.SZ","688151.SH","600319.SH","605033.SH"]

CSI300_RET = {"20260601":-0.98,"20260602":1.45,"20260603":0.49,"20260604":-0.69,
              "20260605":-1.79,"20260608":-2.14,"20260609":1.87,"20260610":-1.11,
              "20260611":-0.55,"20260612":1.16}

CSI5D = {"20260601":-1.57,"20260602":-1.57,"20260603":0.64,
         "20260604":-0.17,"20260605":-1.51,"20260608":0,"20260609":0,
         "20260610":0,"20260611":0}

SIGNALS = ["20260601","20260602","20260603","20260604","20260605",
           "20260608","20260609","20260610","20260611"]

LINE_COLORS = {"实际执行":"#EF5350","N=5 K=3":"#42A5F5","N=10 K=3":"#66BB6A"}


def run_strategy(scores, ret_lk, ad, name, N_base, K):
    """Run backtest for a strategy. Returns DataFrame with [return_date, port_ret]."""
    held = set()
    rows = []

    for i, sd in enumerate(SIGNALS):
        try:
            rd = ad[ad.index(sd)+1]
        except (ValueError, IndexError):
            break
        sc = scores[scores["trade_date"]==sd].set_index("ts_code")["score"]
        dr = ret_lk.get(rd, pd.Series(dtype=float))

        N = N_base
        csi5d = CSI5D.get(sd, 0)
        if csi5d < -1.0:
            N = max(1, int(N * 0.8))

        if name == "实际执行" and i == 0:
            eff = set(JUNE1) & set(dr.index)
            pr = dr[list(eff)].mean() if eff else 0.0
            cost = (0.00076+0.00026)*len(eff)/len(eff)
            held = eff
        elif name == "实际执行" and i == 1:
            N = 5
            top = set(sc.nlargest(N).index) & set(dr.index)
            pr = dr[list(top)].mean() if top else 0.0
            cost = (0.00076+0.00026)*len(top)/len(top)
            held = top
        elif i == 0:
            top = set(sc.nlargest(N).index) & set(dr.index)
            pr = dr[list(top)].mean() if top else 0.0
            cost = (0.00076+0.00026)*len(top)/len(top)
            held = top
        else:
            hv = held & set(dr.index)
            pr = dr[list(hv)].mean() if hv else 0.0
            top = set(sc.nlargest(N).index) & set(dr.index)
            rk = sorted(hv, key=lambda x: sc.get(x,-1e9))
            ts = set(rk[:K]) & hv
            tb = set(sorted(top-hv, key=lambda x: sc.get(x,-1e9), reverse=True)[:K])
            cost = (0.00076+0.00026)*max(len(ts),len(tb))/N
            held = (hv-ts)|tb

        rows.append({"return_date":rd, "port_ret":pr-cost})

    df = pd.DataFrame(rows)
    df["cum"] = (1+df["port_ret"]).cumprod()
    return df


def main():
    scores = pd.read_parquet(PARQUET)
    scores["trade_date"] = scores["trade_date"].astype(str)
    ret_all = pd.read_parquet(PARQUET_ALL)
    ret_all["trade_date"] = ret_all["trade_date"].astype(str)
    rlk = {d: g.set_index("ts_code")["pct_chg"]/100.0 for d,g in ret_all.groupby("trade_date")}
    ad = sorted(ret_all["trade_date"].unique())

    # Run three strategies
    strategies = {}
    for name, N, K in [("实际执行",5,3), ("N=5 K=3",5,3), ("N=10 K=3",10,3)]:
        df = run_strategy(scores, rlk, ad, name, N, K)
        strategies[name] = df
        print(f"[{name}] final: {(df['cum'].iloc[-1]-1)*100:+.2f}%")

    df_actual = strategies["实际执行"]
    cc = (1+np.array([CSI300_RET.get(d,0.0)/100.0 for d in df_actual["return_date"]])).cumprod()
    x = range(len(df_actual))

    # ── Plot ──
    fig, (ax1, ax2) = plt.subplots(2,1,figsize=(13,7.5),sharex=True,
                                    gridspec_kw={"height_ratios":[1.2,1.3]})
    fig.subplots_adjust(top=0.92)

    # Upper: 3 strategy lines
    for name, df in strategies.items():
        dp = df["port_ret"].values*100
        ax1.plot(x, dp, "o-", color=LINE_COLORS[name], linewidth=1.8, markersize=6,
                 label=name, zorder=3)
    ax1.axhline(y=0, color="black", linewidth=0.5)
    ax1.set_ylabel("日收益率 (%)", fontsize=12)
    ax1.set_title("模拟交易竞赛期 — 多策略日收益率对比 (2026年6月1–12日)", fontsize=14, fontweight="bold")
    ax1.legend(fontsize=10, loc="upper left")
    ax1.grid(alpha=0.2)

    # Lower: actual strategy cumulative vs CSI300
    ax2.plot(x, df_actual["cum"].values, "o-", color="#EF5350", linewidth=2.5, markersize=8,
             label="实际执行策略", zorder=3)
    ax2.plot(x, cc, "s--", color="#9E9E9E", linewidth=2, markersize=6,
             label="沪深300基准", zorder=2)
    ax2.axhline(y=1.0, color="black", linewidth=0.5, linestyle="--", alpha=0.3)
    for i,(d,c) in enumerate(zip(df_actual["return_date"],df_actual["cum"])):
        ax2.annotate(f"{c:.4f}", (i,c), textcoords="offset points",
                     xytext=(0,12), fontsize=8, ha="center", fontweight="bold")
    ax2.set_xticks(x)
    ax2.set_xticklabels([f"6/{d[6:8]}" for d in df_actual["return_date"]], fontsize=10)
    ax2.set_ylabel("累计收益 (倍)", fontsize=12)
    ax2.set_xlabel("日期", fontsize=12)
    ax2.grid(alpha=0.2)
    ax2.legend(fontsize=10, loc="upper left")

    p = os.path.join(OUTDIR, "fig_competition.png")
    fig.savefig(p, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[*] {p}")


if __name__ == "__main__":
    main()
