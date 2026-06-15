"""plot_competition_real.py — 按实际成交记录重建竞赛收益

6月1日实际买入10只股票(见交易记录), 后续按N=5 K=3模型轮动.
每日信号→次日开盘买入→当日收盘计算收益.
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

COLORS = {"pos": "#4CAF50", "neg": "#F44336", "v7": "#EF5350", "csi300": "#9E9E9E"}

# 6月1日实际买入的10只股票
JUNE1_HOLDINGS = [
    "003030.SZ", "603029.SH", "300927.SZ", "603686.SH", "688616.SH",
    "603048.SH", "002877.SZ", "688151.SH", "600319.SH", "605033.SH",
]

# CSI300 June returns
JUNE_CSI300 = {
    "20260601": -0.98, "20260602": 1.45, "20260603": 0.49,
    "20260604": -0.69, "20260605": -1.79, "20260608": -2.14,
    "20260609": 1.87, "20260610": -1.11, "20260611": -0.55,
    "20260612": 1.16,
}


def main():
    print("[*] Loading scores ...")
    scores = pd.read_parquet(PARQUET)
    scores["trade_date"] = scores["trade_date"].astype(str)

    print("[*] Loading returns ...")
    ret_all = pd.read_parquet(PARQUET_ALL)
    ret_all["trade_date"] = ret_all["trade_date"].astype(str)

    ret_lookup = {}
    for d, g in ret_all.groupby("trade_date"):
        ret_lookup[d] = g.set_index("ts_code")["pct_chg"] / 100.0

    all_dates = sorted(ret_all["trade_date"].unique())

    # ── 逐日模拟 ──
    # 方案: 每日用模型分数排序选股, 计算当日收盘收益
    # 实际流程: 前一日收盘后跑模型 → 次日开盘买入 → 当日收盘计算收益

    signal_dates = ["20260529", "20260601", "20260602", "20260603",
                    "20260604", "20260605", "20260608", "20260609",
                    "20260610", "20260611"]

    held = set()        # 当前持仓
    daily_rows = []

    for i, signal_d in enumerate(signal_dates):
        # 收益日 = 信号日的下一个交易日
        try:
            next_idx = all_dates.index(signal_d) + 1
            if next_idx >= len(all_dates):
                break
            return_d = all_dates[next_idx]
        except ValueError:
            continue

        day_scores = scores[scores["trade_date"] == signal_d].set_index("ts_code")["score"]
        day_rets = ret_lookup.get(return_d, pd.Series(dtype=float))

        if i == 0:
            # 6月1日: 使用实际买入的10只股票
            effective = set(JUNE1_HOLDINGS) & set(day_rets.index)
            held = effective
            port_ret = day_rets[list(effective)].mean() if effective else 0.0
            cost = (0.00076 + 0.00026) * 10 / 10  # 全仓买入成本
            n_used, k_used = 10, 10
        else:
            # 后续: N=5 K=3 轮动
            N, K = 5, 3
            held_valid = held & set(day_rets.index)
            port_ret = day_rets[list(held_valid)].mean() if held_valid else 0.0

            top_n = set(day_scores.nlargest(N).index) & set(day_rets.index)
            held_ranked = sorted(held_valid, key=lambda x: day_scores.get(x, -1e9))
            to_sell = set(held_ranked[:K]) & held_valid
            candidates = top_n - held_valid
            to_buy = set(sorted(candidates, key=lambda x: day_scores.get(x, -1e9), reverse=True)[:K])
            n_traded = max(len(to_sell), len(to_buy))
            cost = (0.00076 + 0.00026) * n_traded / N
            held = (held_valid - to_sell) | to_buy
            n_used, k_used = N, K

        port_ret_net = port_ret - cost
        daily_rows.append({
            "return_date": return_d,
            "port_ret": port_ret_net,
            "port_ret_gross": port_ret,
        })
        held_list = ", ".join(sorted(held)[:3]) + ("..." if len(held) > 3 else "")
        print(f"  {signal_d}→{return_d}  held={len(held)}  "
              f"ret={port_ret_net*100:+.2f}%  [{held_list}]")

    df = pd.DataFrame(daily_rows)
    if len(df) == 0:
        print("[!] No data"); return

    df["cum"] = (1 + df["port_ret"]).cumprod()

    # CSI300
    csi_returns = [JUNE_CSI300.get(d, 0.0) / 100.0 for d in df["return_date"]]
    csi_cum = (1 + np.array(csi_returns)).cumprod()

    # ── Plot ──
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(13, 7.5), sharex=True,
                                    gridspec_kw={"height_ratios": [1.1, 1.3]})
    fig.subplots_adjust(top=0.92)

    x = range(len(df))
    daily_pct = df["port_ret"].values * 100
    colors_bar = [COLORS["pos"] if r > 0 else COLORS["neg"] for r in daily_pct]

    ax1.bar(x, daily_pct, color=colors_bar, edgecolor="white", width=0.55, zorder=3)
    ax1.axhline(y=0, color="black", linewidth=0.5)
    for i, (d, r) in enumerate(zip(df["return_date"], daily_pct)):
        y = r + (0.4 if r >= 0 else -0.8)
        ax1.text(i, y, f"{r:+.1f}%", ha="center", fontsize=9,
                 fontweight="bold", color=COLORS["pos"] if r >= 0 else COLORS["neg"])
    ax1.set_ylabel("日收益率 (%)", fontsize=12)
    ax1.set_title("模拟交易竞赛期 — 每日收益 (2026年6月1–12日)", fontsize=14, fontweight="bold")
    ax1.grid(axis="y", alpha=0.2)

    ax2.plot(x, df["cum"].values, "o-", color=COLORS["v7"], linewidth=2.5, markersize=8,
             label="V7 Spatial 策略", zorder=3)
    ax2.plot(x, csi_cum, "s--", color=COLORS["csi300"], linewidth=2, markersize=6,
             label="沪深300基准", zorder=2)

    ax2.axhline(y=1.0, color="black", linewidth=0.5, linestyle="--", alpha=0.3)
    for i, (d, c) in enumerate(zip(df["return_date"], df["cum"])):
        ax2.annotate(f"{c:.4f}", (i, c), textcoords="offset points",
                     xytext=(0, 12), fontsize=8, ha="center", fontweight="bold")
    ax2.set_xticks(x)
    ax2.set_xticklabels([f"6/{d[6:8]}" for d in df["return_date"]], fontsize=10)
    ax2.set_ylabel("累计收益 (倍)", fontsize=12)
    ax2.set_xlabel("日期", fontsize=12)
    ax2.grid(alpha=0.2)
    ax2.legend(fontsize=10, loc="upper left")

    path = os.path.join(OUTDIR, "fig_competition.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[*] Saved {path}")

    final = (df["cum"].iloc[-1] - 1) * 100
    csi_final = (csi_cum[-1] - 1) * 100
    print(f"[*] 策略累计: {final:+.2f}%  沪深300: {csi_final:+.2f}%")


if __name__ == "__main__":
    main()
