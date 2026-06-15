#!/usr/bin/env python3
"""
图表生成脚本 — A股股票排序预测实验报告
生成所有实验数据的可视化图表，输出至 figures/ 目录
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import os

# ── 全局设置 ─────────────────────────────────────────────
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["SimHei", "DejaVu Sans", "Arial"],
    "axes.unicode_minus": False,
    "figure.dpi": 150,
    "savefig.dpi": 150,
    "savefig.bbox": "tight",
    "figure.facecolor": "white",
})

OUTDIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "figures")
os.makedirs(OUTDIR, exist_ok=True)

COLORS = {
    "gru": "#2196F3", "tf": "#FF9800", "mlp": "#4CAF50",
    "spatial": "#E91E63", "baseline": "#607D8B", "v7": "#E91E63",
    "v8": "#795548", "csi300": "#9E9E9E", "ew": "#BDBDBD",
    "pos": "#4CAF50", "neg": "#F44336",
    "alpha": "#2196F3", "beta": "#FF9800",
    "listmle": "#4CAF50", "weighted": "#F44336", "lambdarank": "#FF9800",
}
MARKERS = ["o", "s", "D", "^", "v", "<", ">", "p", "*", "h"]


def save(fig, name):
    path = os.path.join(OUTDIR, name)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  [OK] {name}")


# ═══════════════════════════════════════════════════════════
# 图1: V2三模型 Val IC 对比柱状图
# ═══════════════════════════════════════════════════════════
def fig1_model_comparison():
    models = ["GRU\n(H=128,L=1)", "Transformer\n(d=96,heads=4)", "MLP\n(4×1024)"]
    ics = [0.1029, 0.1009, 0.0843]
    colors = [COLORS["gru"], COLORS["tf"], COLORS["mlp"]]

    fig, ax = plt.subplots(figsize=(7, 5))
    bars = ax.bar(models, ics, color=colors, edgecolor="white", width=0.55, zorder=3)
    ymax = max(ics)
    for bar, ic in zip(bars, ics):
        ax.text(bar.get_x() + bar.get_width() / 2, ic + ymax * 0.03, f"{ic:.4f}",
                ha="center", va="bottom", fontsize=12, fontweight="bold")

    ax.set_ylabel("Validation Rank IC", fontsize=12)
    ax.set_title("V2 多模型架构 Val IC 对比 (22-dim features)", fontsize=14, fontweight="bold")
    ax.set_ylim(0, ymax * 1.18)
    ax.grid(axis="y", alpha=0.3, zorder=0)
    ax.axhline(y=0.1029, color=COLORS["gru"], linestyle="--", alpha=0.5, linewidth=1, label=f"GRU best = 0.1029")
    ax.legend(fontsize=9, loc="upper right")
    save(fig, "fig1_model_comparison.png")


# ═══════════════════════════════════════════════════════════
# 图2: 版本演化 Val/Test IC
# ═══════════════════════════════════════════════════════════
def fig2_version_ic_evolution():
    versions = ["V1\nGRU", "V2\n22-dim", "V5\n26-dim Fix", "V6\nSpatial", "V7\nGRU T+1", "V7\nSpatial T+1"]
    val_ic = [0.1042, 0.1029, 0.1114, 0.1134, 0.1023, 0.1062]
    test_ic = [None, 0.042, 0.053, 0.051, 0.052, 0.048]
    # Adjust test IC to same scale (none for V1)
    test_ic_adj = [np.nan, 0.042, 0.053, 0.051, 0.052, 0.048]
    # Normalize: test IC × 2 for visibility on same scale
    test_ic_scaled = [np.nan, 0.042*2, 0.053*2, 0.051*2, 0.052*2, 0.048*2]

    fig, ax1 = plt.subplots(figsize=(10, 5.5))
    x = np.arange(len(versions))
    w = 0.35
    ymax = max(max(val_ic), max([v for v in test_ic_scaled if not np.isnan(v)]))

    bars1 = ax1.bar(x - w/2, val_ic, w, label="Val IC (T+5 for V5-V6, T+1 for V7)", color=COLORS["gru"], edgecolor="white", zorder=3)
    bars2 = ax1.bar(x + w/2, test_ic_scaled, w, label="Test T+1 IC (×2 scale)", color=COLORS["spatial"], edgecolor="white", zorder=3)

    # Add labels
    for bar, ic in zip(bars1, val_ic):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + ymax * 0.02, f"{ic:.4f}",
                ha="center", va="bottom", fontsize=8, fontweight="bold")
    for bar, ic_orig in zip(bars2, test_ic_adj):
        if not np.isnan(ic_orig):
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + ymax * 0.02, f"{ic_orig:.3f}",
                    ha="center", va="bottom", fontsize=8, fontweight="bold", color=COLORS["spatial"])

    ax1.set_ylabel("Rank IC", fontsize=12)
    ax1.set_title("各版本 Val/Test IC 演化", fontsize=14, fontweight="bold")
    ax1.set_xticks(x)
    ax1.set_xticklabels(versions, fontsize=9)
    ax1.legend(loc="upper right", fontsize=9)
    ax1.grid(axis="y", alpha=0.3, zorder=0)
    ax1.set_ylim(0, ymax * 1.22)

    # Add vertical line separating T+5 and T+1
    ax1.axvline(x=3.5, color="red", linestyle="--", alpha=0.4, linewidth=1)
    ax1.text(3.5, ymax * 1.05, "T+5→T+1\nLabel Switch", ha="center", fontsize=7, color="red", alpha=0.7)
    save(fig, "fig2_version_ic_evolution.png")


# ═══════════════════════════════════════════════════════════
# 图3: Strategy A/B/C 对比
# ═══════════════════════════════════════════════════════════
def fig3_strategy_comparison():
    strategies = ["Baseline\n(Always 100%)", "Strategy A\n(Vol Risk-Off)", "Strategy B\n(Momentum Stop)", "Strategy B\n+Dispersion"]
    returns = [2.34, 5.58, 6.85, 7.67]
    sharpes = [0.35, 0.79, 0.93, 1.05]
    colors_list = [COLORS["baseline"], COLORS["tf"], COLORS["pos"], COLORS["spatial"]]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    ymax_r = max(returns)
    ymax_s = max(sharpes)

    bars = ax1.bar(strategies, returns, color=colors_list, edgecolor="white", width=0.5, zorder=3)
    for bar, r in zip(bars, returns):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + ymax_r * 0.04, f"+{r:.2f}%",
                ha="center", fontsize=11, fontweight="bold")
    ax1.set_ylabel("Cumulative Return (%)", fontsize=11)
    ax1.set_title("Strategy Comparison — Return (Feb-May 2026)", fontsize=12, fontweight="bold")
    ax1.grid(axis="y", alpha=0.3, zorder=0)
    ax1.tick_params(axis="x", labelsize=8)
    ax1.set_ylim(0, ymax_r * 1.18)

    bars2 = ax2.bar(strategies, sharpes, color=colors_list, edgecolor="white", width=0.5, zorder=3)
    for bar, s in zip(bars2, sharpes):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + ymax_s * 0.04, f"{s:.2f}",
                ha="center", fontsize=11, fontweight="bold")
    ax2.set_ylabel("Sharpe Ratio", fontsize=11)
    ax2.set_title("Strategy Comparison — Sharpe Ratio", fontsize=12, fontweight="bold")
    ax2.grid(axis="y", alpha=0.3, zorder=0)
    ax2.tick_params(axis="x", labelsize=8)
    ax2.set_ylim(0, ymax_s * 1.18)

    fig.suptitle("V3 Strategy Backtesting Results", fontsize=14, fontweight="bold", y=1.02)
    save(fig, "fig3_strategy_comparison.png")


# ═══════════════════════════════════════════════════════════
# 图4: T=30/60/90 IC对比
# ═══════════════════════════════════════════════════════════
def fig4_window_comparison():
    windows = ["T=30", "T=60", "T=90"]
    ics = [0.0363, 0.0424, 0.0430]
    ics_pos = [61, 64, 65]
    rank_corrs = [0.845, 1.0, 0.9876]

    fig, ax1 = plt.subplots(figsize=(8, 5))
    x = np.arange(len(windows))
    w = 0.35
    ymax = max(ics)

    bars = ax1.bar(x, ics, w, color=[COLORS["tf"], COLORS["gru"], COLORS["baseline"]], edgecolor="white", zorder=3)
    for bar, ic in zip(bars, ics):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + ymax * 0.03, f"IC={ic:.4f}",
                ha="center", fontsize=10, fontweight="bold")
    ax1.set_ylabel("Mean Rank IC", fontsize=12)
    ax1.set_xticks(x)
    ax1.set_xticklabels(windows, fontsize=11)
    ax1.set_title("Multi-Window GRU IC Comparison (Test Period)", fontsize=14, fontweight="bold")
    ax1.grid(axis="y", alpha=0.3, zorder=0)
    ax1.set_ylim(0, ymax * 1.2)

    ax2 = ax1.twinx()
    ax2.plot(x, rank_corrs, "o-", color=COLORS["spatial"], linewidth=2, markersize=10, label="Score Rank Corr (vs T=60)")
    for i, rc in enumerate(rank_corrs):
        ax2.annotate(f"r={rc:.2f}", (x[i], rank_corrs[i]), textcoords="offset points",
                    xytext=(0, 15), ha="center", fontsize=9, color=COLORS["spatial"], fontweight="bold")
    ax2.set_ylabel("Score Rank Correlation (vs T=60)", fontsize=11, color=COLORS["spatial"])
    ax2.set_ylim(0.7, 1.15)
    ax2.legend(loc="lower left", fontsize=9)
    ax2.axhline(y=0.9, color="gray", linestyle="--", alpha=0.5, label="r=0.9 threshold")
    save(fig, "fig4_window_comparison.png")


# ═══════════════════════════════════════════════════════════
# 图5: KNN proxy K-α IC增益热力图
# ═══════════════════════════════════════════════════════════
def fig5_knn_proxy_heatmap():
    K_vals = [3, 5, 10, 20]
    alpha_vals = [0.1, 0.3, 0.5]
    ic_data = np.array([
        [0.0431, 0.0453, 0.0466],  # K=3
        [0.0430, 0.0455, 0.0472],  # K=5
        [0.0430, 0.0460, 0.0474],  # K=10
        [0.0428, 0.0448, 0.0466],  # K=20
    ])
    baseline = 0.0424
    ic_gain = ic_data - baseline

    fig, ax = plt.subplots(figsize=(8, 5))
    im = ax.imshow(ic_gain, cmap="RdYlGn", aspect="auto", vmin=-0.001, vmax=0.006)

    for i in range(len(K_vals)):
        for j in range(len(alpha_vals)):
            ax.text(j, i, f"{ic_data[i,j]:.4f}\n(+{ic_gain[i,j]:.4f})",
                   ha="center", va="center", fontsize=10, fontweight="bold")

    ax.set_xticks(range(len(alpha_vals)))
    ax.set_xticklabels([f"α={a}" for a in alpha_vals], fontsize=11)
    ax.set_yticks(range(len(K_vals)))
    ax.set_yticklabels([f"K={k}" for k in K_vals], fontsize=11)
    ax.set_title(f"KNN Proxy Spatial Validation — IC Heatmap (Baseline IC={baseline:.4f})",
                fontsize=13, fontweight="bold")
    fig.colorbar(im, ax=ax, label="IC Gain over Baseline", shrink=0.8)
    save(fig, "fig5_knn_proxy_heatmap.png")


# ═══════════════════════════════════════════════════════════
# 图6: V1→V2空间注意力修正效果对比
# ═══════════════════════════════════════════════════════════
def fig6_spatial_v1_v2():
    configs = ["Baseline\n(GRU only)", "V1 (d=128)\nno proj", "V2 (d=32)\nresidual", "V2 (d=32)\nconcat", "V2 (d=32)\ngated"]
    ics = [0.1114, 0.1086, 0.1131, 0.1126, 0.1122]
    changes = [0, -0.0028, +0.0017, +0.0012, +0.0008]
    colors_list = [COLORS["baseline"], COLORS["neg"], COLORS["pos"], COLORS["gru"], COLORS["tf"]]

    fig, ax = plt.subplots(figsize=(9, 5.5))
    x = np.arange(len(configs))
    ymax = max(ics)
    bars = ax.bar(x, ics, color=colors_list, edgecolor="white", width=0.55, zorder=3)

    for bar, ic, ch in zip(bars, ics, changes):
        color = COLORS["pos"] if ch > 0 else (COLORS["neg"] if ch < 0 else "black")
        label = f"{ic:.4f}"
        if ch != 0:
            label += f"\n({'↑' if ch>0 else '↓'}{abs(ch):.4f})"
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + ymax * 0.02,
               label, ha="center", fontsize=9, fontweight="bold", color=color)

    ax.axhline(y=0.1114, color=COLORS["baseline"], linestyle="--", alpha=0.4, label="Baseline (0.1114)")
    ax.set_xticks(x)
    ax.set_xticklabels(configs, fontsize=9)
    ax.set_ylabel("Validation Rank IC", fontsize=12)
    ax.set_title("Spatial Attention V1→V2 修正效果 (d=32 bottleneck 是关键)", fontsize=14, fontweight="bold")
    ax.legend(fontsize=9)
    ax.grid(axis="y", alpha=0.3, zorder=0)
    ax.set_ylim(0.104, ymax * 1.06)
    save(fig, "fig6_spatial_v1_v2.png")


# ═══════════════════════════════════════════════════════════
# 图7: V6 Phase 2 K值扫描
# ═══════════════════════════════════════════════════════════
def fig7_k_sweep():
    K_vals = [5, 10, 20]
    concat_ic = [0.1134, 0.1126, 0.1111]
    residual_ic = [0.1124, 0.1131, 0.1129]

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(K_vals, concat_ic, "o-", color=COLORS["gru"], linewidth=2, markersize=10, label="Concat Fusion")
    ax.plot(K_vals, residual_ic, "s--", color=COLORS["spatial"], linewidth=2, markersize=10, label="Residual Fusion")

    for k, ci, ri in zip(K_vals, concat_ic, residual_ic):
        ax.annotate(f"{ci:.4f}", (k, ci), textcoords="offset points", xytext=(8, -12),
                   fontsize=10, color=COLORS["gru"], fontweight="bold")
        ax.annotate(f"{ri:.4f}", (k, ri), textcoords="offset points", xytext=(8, 10),
                   fontsize=10, color=COLORS["spatial"], fontweight="bold")

    ax.axhline(y=0.1114, color=COLORS["baseline"], linestyle="--", alpha=0.4, label="Baseline GRU (0.1114)")
    ax.set_xlabel("K (Number of Neighbors)", fontsize=12)
    ax.set_ylabel("Validation Rank IC", fontsize=12)
    ax.set_title("Spatial Attention K值扫描 (d=32, N=1024)", fontsize=14, fontweight="bold")
    ax.legend(fontsize=10)
    ax.grid(alpha=0.3)
    ax.set_xticks(K_vals)
    save(fig, "fig7_k_sweep.png")


# ═══════════════════════════════════════════════════════════
# 图8: 按市值分位IC
# ═══════════════════════════════════════════════════════════
def fig8_cap_ic():
    caps = ["大市值\n(top 1/3)", "中等市值", "小市值\n(bot 1/3)"]
    ics = [0.044, 0.057, 0.060]
    ic_pos = [57.5, 63.0, 68.5]

    fig, ax1 = plt.subplots(figsize=(7, 5))
    x = np.arange(len(caps))
    w = 0.35
    ymax = max(ics)

    bars = ax1.bar(x - w/2, ics, w, color=[COLORS["gru"], COLORS["tf"], COLORS["spatial"]], edgecolor="white", zorder=3)
    for bar, ic in zip(bars, ics):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + ymax * 0.03, f"IC={ic:.3f}",
                ha="center", fontsize=11, fontweight="bold")
    ax1.set_ylabel("Test Rank IC", fontsize=12)
    ax1.set_xticks(x)
    ax1.set_xticklabels(caps, fontsize=10)
    ax1.set_title("V6 Spatial — IC by Market Cap Quantile (Feb-May 2026)", fontsize=13, fontweight="bold")
    ax1.grid(axis="y", alpha=0.3, zorder=0)
    ax1.set_ylim(0, ymax * 1.2)

    ax2 = ax1.twinx()
    ax2.bar(x + w/2, ic_pos, w, color=["#90CAF9", "#FFE0B2", "#F48FB1"], edgecolor="white", alpha=0.6, zorder=3)
    for bar, pos in zip([None]*3, ic_pos):  # labels on ax2 bars
        pass
    ax2.set_ylabel("IC > 0 (%)", fontsize=12)
    ax2.set_ylim(50, 75)
    for i, pos in enumerate(ic_pos):
        ax2.text(i + w/2, pos + 0.5, f"{pos}%", ha="center", fontsize=9, color="#555")

    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor=COLORS["gru"], label="Large Cap"),
        Patch(facecolor=COLORS["tf"], label="Mid Cap"),
        Patch(facecolor=COLORS["spatial"], label="Small Cap"),
    ]
    ax1.legend(handles=legend_elements, loc="upper left", fontsize=9)
    save(fig, "fig8_cap_ic.png")


# ═══════════════════════════════════════════════════════════
# 图9: V7 N×K扫参热力图
# ═══════════════════════════════════════════════════════════
def fig9_nk_sweep_heatmap():
    N_vals = [5, 6, 7, 8, 9, 10]
    K_vals = [2, 3, 5]
    # Data: net return in % — from V7 spatial sweep
    data = np.array([
        [10.81, 15.45, np.nan],   # N=5
        [np.nan, 13.19, np.nan],  # N=6  (K=2,3 from fine sweep; K=5 not tested)
        [np.nan, 10.88, np.nan],  # N=7
        [np.nan, 9.68, np.nan],   # N=8
        [np.nan, 6.76, np.nan],   # N=9
        [np.nan, 4.69, 4.46],     # N=10
    ])
    mask = np.isnan(data)

    fig, ax = plt.subplots(figsize=(8, 6))
    cmap = plt.cm.RdYlGn
    im = ax.imshow(data, cmap=cmap, aspect="auto", vmin=0, vmax=16)

    for i in range(len(N_vals)):
        for j in range(len(K_vals)):
            if not mask[i, j]:
                ax.text(j, i, f"+{data[i,j]:.1f}%", ha="center", va="center",
                       fontsize=12, fontweight="bold", color="black")

    ax.set_xticks(range(len(K_vals)))
    ax.set_xticklabels([f"K={k}" for k in K_vals], fontsize=12)
    ax.set_yticks(range(len(N_vals)))
    ax.set_yticklabels([f"N={n}" for n in N_vals], fontsize=12)
    ax.set_title("V7 Spatial N×K Strategy Sweep — Net Return (%)\n(600 stocks, Feb-May 2026, Strategy B)", fontsize=13, fontweight="bold")
    fig.colorbar(im, ax=ax, label="Cumulative Net Return (%)", shrink=0.85)
    save(fig, "fig9_nk_sweep_heatmap.png")


# ═══════════════════════════════════════════════════════════
# 图10: 月度收益分解
# ═══════════════════════════════════════════════════════════
def fig10_monthly_decomposition():
    months = ["Jan", "Feb", "Mar", "Apr", "May"]
    net_returns = [56.40, -2.32, 4.84, 3.68, 3.24]
    colors_list = [COLORS["pos"] if r > 0 else COLORS["neg"] for r in net_returns]
    contributions = [85.7, 0, 7.4, 5.6, 4.9]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    ymax_r = max(net_returns)
    ymin_r = min(net_returns)

    bars = ax1.bar(months, net_returns, color=colors_list, edgecolor="white", width=0.55, zorder=3)
    for bar, r in zip(bars, net_returns):
        y = bar.get_height() + (ymax_r * 0.06 if r > 0 else ymin_r * 0.06)
        ax1.text(bar.get_x() + bar.get_width()/2, y, f"{r:+.1f}%",
                ha="center", fontsize=11, fontweight="bold",
                color=COLORS["pos"] if r > 0 else COLORS["neg"])
    ax1.set_ylabel("Net Return (%)", fontsize=11)
    ax1.set_title("Monthly Net Return (N=5, K=3, th=-1.0%)", fontsize=12, fontweight="bold")
    ax1.grid(axis="y", alpha=0.3, zorder=0)
    ax1.axhline(y=0, color="black", linewidth=0.5)
    ax1.set_ylim(ymin_r * 1.3, ymax_r * 1.2)

    # Pie chart for contributions (positive months only)
    pos_months = ["Jan\n85.7%", "Mar\n7.4%", "Apr\n5.6%", "May\n4.9%"]
    pos_vals = [85.7, 7.4, 5.6, 4.9]
    pie_colors = ["#1565C0", "#42A5F5", "#90CAF9", "#BBDEFB"]
    ax2.pie(pos_vals, labels=pos_months, colors=pie_colors, autopct="",
           startangle=90, textprops={"fontsize": 10})
    ax2.set_title("Profit Contribution by Month\n(Total: +65.83%)", fontsize=12, fontweight="bold")

    fig.suptitle("V7 月度收益分解 (Jan-May 2026)", fontsize=14, fontweight="bold", y=1.02)
    save(fig, "fig10_monthly_decomposition.png")


# ═══════════════════════════════════════════════════════════
# 图11: 累计收益曲线 (v7 Spatial vs CSI300)
# ═══════════════════════════════════════════════════════════
def fig11_cumulative_returns():
    # Approximate daily cumulative returns reconstructed from known milestones
    # Window data: 10 windows of ~9.4 days each
    cum_vals = [0, 37.05, 56.39, 55.09, 56.94, 56.60, 57.66, 61.45, 65.10, 66.73, 65.83]
    # Day markers for x-axis
    day_marks = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 94]
    # CSI300 approximation: +6.22% total over 94 days
    csi300 = [0, 1.1, 1.3, -5.0, -1.5, -2.0, -0.8, 2.5, 5.5, 7.0, 6.22]

    fig, ax = plt.subplots(figsize=(11, 5.5))
    ax.plot(day_marks, cum_vals, "-", color=COLORS["spatial"], linewidth=2.5, marker="o",
           markersize=6, label="v7 Spatial (N=5,K=3,th=-1.0%)", zorder=3)
    ax.plot(day_marks, csi300, "-", color=COLORS["csi300"], linewidth=2, marker="s",
           markersize=5, label="CSI300 Index", zorder=2)

    # Annotations
    ax.annotate(f"+65.83%", (94, 65.83), textcoords="offset points", xytext=(10, 5),
               fontsize=12, fontweight="bold", color=COLORS["spatial"])
    ax.annotate(f"+6.22%", (94, 6.22), textcoords="offset points", xytext=(10, -5),
               fontsize=10, color=COLORS["csi300"])
    ax.annotate("Excess: +59.61%", (50, 40), fontsize=10, color="green", fontweight="bold",
               ha="center")

    ax.set_xlabel("Trading Days (Jan 5 – May 29, 2026)", fontsize=12)
    ax.set_ylabel("Cumulative Return (%)", fontsize=12)
    ax.set_title("V7 Spatial vs CSI300 — Cumulative Return (94 days, net of costs)", fontsize=14, fontweight="bold")
    ax.legend(fontsize=11, loc="upper left")
    ax.grid(alpha=0.3)
    ax.axhline(y=0, color="black", linewidth=0.5)
    save(fig, "fig11_cumulative_returns.png")


# ═══════════════════════════════════════════════════════════
# 图12: V8 vs V7 对比
# ═══════════════════════════════════════════════════════════
def fig12_v8_vs_v7():
    metrics = ["Val IC", "Test T+1 IC", "Test IC>0%", "回测 Net%"]
    v7_vals = [0.1062, 0.048, 64.4, 65.83]
    v8_vals = [0.1037, 0.044, 63.0, 1.80]

    fig, axes = plt.subplots(1, 4, figsize=(14, 4.5))

    for i, (ax, metric, v7, v8) in enumerate(zip(axes, metrics, v7_vals, v8_vals)):
        colors_list = [COLORS["v7"], COLORS["v8"]]
        ymax = max(v7, v8)
        bars = ax.bar(["v7", "v8"], [v7, v8], color=colors_list, edgecolor="white", width=0.5, zorder=3)
        for bar, val in zip(bars, [v7, v8]):
            fmt = f"{val:.3f}" if isinstance(val, float) and val < 10 else f"{val:.2f}"
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + ymax * 0.06,
                   fmt, ha="center", fontsize=10, fontweight="bold")
        ax.set_title(metric, fontsize=11, fontweight="bold")
        ax.grid(axis="y", alpha=0.3, zorder=0)
        ax.set_ylim(0, ymax * 1.25)
        delta = v8 - v7
        color = COLORS["neg"] if delta < 0 else COLORS["pos"]
        arrow = "↓" if delta < 0 else "↑"
        ax.text(0.5, ymax * 0.92, f"{arrow}{abs(delta):.3f}", ha="center", fontsize=9,
                color=color, fontweight="bold", transform=ax.get_xaxis_transform())

    fig.suptitle("V8 (31-dim + Industry Emb) vs V7 (26-dim) — 全面退步",
                fontsize=14, fontweight="bold", y=1.02)
    save(fig, "fig12_v8_vs_v7.png")


# ═══════════════════════════════════════════════════════════
# 图13: 三种Loss对比
# ═══════════════════════════════════════════════════════════
def fig13_loss_comparison():
    losses = ["ListMLE", "Weighted\nListMLE (α=0.9)", "LambdaRank"]
    ics = [0.1037, -0.0670, 0.0069]
    colors_list = [COLORS["listmle"], COLORS["weighted"], COLORS["lambdarank"]]

    fig, ax = plt.subplots(figsize=(8, 5))
    x = np.arange(len(losses))
    ymax = max(ics)
    ymin = min(ics)
    bars = ax.bar(x, ics, color=colors_list, edgecolor="white", width=0.5, zorder=3)

    for bar, ic in zip(bars, ics):
        y = bar.get_height() + (ymax * 0.04 if ic > 0 else ymax * 0.04)
        ax.text(bar.get_x() + bar.get_width()/2, y, f"{ic:.4f}",
               ha="center", fontsize=12, fontweight="bold",
               color=COLORS["pos"] if ic > 0.05 else (COLORS["neg"] if ic < 0 else "black"))

    ax.axhline(y=0, color="black", linewidth=0.5)
    ax.axhline(y=0.1037, color=COLORS["listmle"], linestyle="--", alpha=0.5, label="ListMLE = 0.1037")
    ax.set_xticks(x)
    ax.set_xticklabels(losses, fontsize=10)
    ax.set_ylabel("Validation Rank IC", fontsize=12)
    ax.set_title("V8 损失函数对比 (31-dim, 完全相同的训练条件)", fontsize=14, fontweight="bold")
    ax.legend(fontsize=9)
    ax.grid(axis="y", alpha=0.3, zorder=0)
    ax.set_ylim(ymin * 1.3, ymax * 1.2)

    # Red X over failed losses
    ax.annotate("X", (1, -0.01), fontsize=30, color=COLORS["neg"], ha="center", alpha=0.6, fontweight="bold")
    ax.annotate("X", (2, 0.05), fontsize=30, color=COLORS["neg"], ha="center", alpha=0.6, fontweight="bold")
    save(fig, "fig13_loss_comparison.png")


# ═══════════════════════════════════════════════════════════
# 图14: 特征IC诊断排名
# ═══════════════════════════════════════════════════════════
def fig14_feature_ic_ranking():
    features = [
        "mf_flow_hhi", "amihud_20", "mf_sm_lg_div", "streak", "ret_kurt_20",
        "max_dd_20", "ret", "mf_net_vol_pct", "vwap_gap", "circ_mv",
        "mf_lg_net_pct", "pe", "total_mv", "ret_skew_20", "price_pos_20",
    ]
    ics = [
        0.0464, 0.0284, 0.0091, 0.0015, 0.0003,
        -0.0040, -0.0068, -0.0078, -0.0140, -0.0142,
        -0.0154, -0.0206, -0.0221, -0.0239, -0.0361,
    ]
    colors_list = [COLORS["pos"] if ic > 0 else COLORS["neg"] for ic in ics]

    fig, ax = plt.subplots(figsize=(10, 6))
    y_pos = range(len(features))
    xmax = max(abs(v) for v in ics)
    bars = ax.barh(y_pos, ics, color=colors_list, edgecolor="white", height=0.7, zorder=3)

    for bar, ic in zip(bars, ics):
        offset = xmax * 0.03
        x = bar.get_width() + (offset if ic > 0 else -xmax * 0.18)
        ax.text(x, bar.get_y() + bar.get_height()/2, f"{ic:+.4f}",
               va="center", fontsize=9, fontweight="bold",
               color=COLORS["pos"] if ic > 0 else COLORS["neg"])

    ax.set_yticks(y_pos)
    ax.set_yticklabels(features, fontsize=9)
    ax.invert_yaxis()
    ax.set_xlabel("Mean Rank IC (2024-2025, 338 days)", fontsize=12)
    ax.set_title("V7 Feature IC Diagnosis — Single-Factor Predictive Power (Top 15)",
                fontsize=13, fontweight="bold")
    ax.axvline(x=0, color="black", linewidth=0.5)
    ax.grid(axis="x", alpha=0.3, zorder=0)
    ax.set_xlim(-xmax * 1.35, xmax * 1.15)
    save(fig, "fig14_feature_ic_ranking.png")


# ═══════════════════════════════════════════════════════════
# 图15: V9 λ-sweep
# ═══════════════════════════════════════════════════════════
def fig15_lambda_sweep():
    lambdas = [0, 0.5, 1.0, 2.0, 3.0, 5.0, 10.0]
    r_vol = [65.83, 65.45, 58.19, 61.00, 60.50, 59.80, 59.00]
    bear_vol = [65.83, 65.73, 58.46, 60.50, 60.00, 59.30, 58.50]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(lambdas, r_vol, "o-", color=COLORS["gru"], linewidth=2, markersize=8, label="r + g(regime)×λ×σ")
    ax.plot(lambdas, bear_vol, "s--", color=COLORS["spatial"], linewidth=2, markersize=8, label="bear_pure_vol")

    ax.axhline(y=65.83, color="gray", linestyle="--", alpha=0.4, linewidth=1, label="Baseline (v7, λ=0): +65.83%")
    ax.fill_between(lambdas, 65.83, 66.0, alpha=0.1, color="green")

    ax.annotate("any λ > 0\n≤ baseline", (3, 59), fontsize=16, color=COLORS["neg"],
               ha="center", fontweight="bold", alpha=0.5)

    ax.set_xlabel("λ (Vol signal weight)", fontsize=12)
    ax.set_ylabel("Cumulative Net Return (%)", fontsize=12)
    ax.set_title("V9 Phase 0 — Sell Model λ-Sweep (Regime-Conditional Vol)", fontsize=14, fontweight="bold")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)
    save(fig, "fig15_lambda_sweep.png")


# ═══════════════════════════════════════════════════════════
# 图16: Jan效应 Alpha/Beta分解
# ═══════════════════════════════════════════════════════════
def fig16_jan_decomposition():
    periods = ["Jan 2026\n(Profit)", "Feb-May 2026\n(Loss)"]
    alpha_contrib = [29.6, -39.0]
    beta_contrib = [3.4, 9.2]
    total = [33.0, -30.8]

    fig, ax = plt.subplots(figsize=(8, 5.5))
    x = np.arange(len(periods))
    w = 0.45

    ax.bar(x, beta_contrib, w, color=COLORS["beta"], edgecolor="white", label="β × Market", zorder=3)
    ax.bar(x, alpha_contrib, w, bottom=beta_contrib, color=COLORS["alpha"], edgecolor="white", label="α (Stock Selection)", zorder=3)

    # Net labels
    for i, (t, a, b) in enumerate(zip(total, alpha_contrib, beta_contrib)):
        y = t + (3 if t > 0 else -5)
        ax.text(i, y, f"Net: {t:+.1f}%", ha="center", fontsize=13, fontweight="bold",
               color=COLORS["pos"] if t > 0 else COLORS["neg"])
        ax.text(i, b/2, f"β: {b:+.1f}%", ha="center", fontsize=10, color="white", fontweight="bold")
        ax.text(i, b + a/2, f"α: {a:+.1f}%", ha="center", fontsize=10, color="white", fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels(periods, fontsize=11)
    ax.set_ylabel("Cumulative Contribution (%)", fontsize=12)
    ax.set_title("Jan 2026 Alpha Decomposition — 88% from True Alpha", fontsize=14, fontweight="bold")
    ax.legend(fontsize=10, loc="lower left")
    ax.axhline(y=0, color="black", linewidth=0.5)
    ax.grid(axis="y", alpha=0.3, zorder=0)
    save(fig, "fig16_jan_decomposition.png")


# ═══════════════════════════════════════════════════════════
# 图17: 月度IC趋势
# ═══════════════════════════════════════════════════════════
def fig17_monthly_ic_trend():
    months = ["Jan", "Feb", "Mar", "Apr", "May"]
    ics = [0.124, 0.066, 0.117, -0.007, 0.013]
    p_values = [0.000, 0.000, 0.000, 0.490, 0.186]
    colors_list = [COLORS["pos"] if ic > 0 else COLORS["neg"] for ic in ics]
    significance = ["***", "***", "***", "ns", "ns"]

    fig, ax = plt.subplots(figsize=(9, 5))
    x = np.arange(len(months))
    ymax = max(ics)
    ymin = min(ics)
    bars = ax.bar(x, ics, color=colors_list, edgecolor="white", width=0.55, zorder=3)

    for bar, ic, sig in zip(bars, ics, significance):
        y = bar.get_height() + (ymax * 0.08 if ic > 0 else ymin * 0.15)
        ax.text(bar.get_x() + bar.get_width()/2, y, f"{ic:+.3f} {sig}",
               ha="center", fontsize=11, fontweight="bold",
               color=COLORS["pos"] if ic > 0 else COLORS["neg"])

    ax.axhline(y=0, color="black", linewidth=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels(months, fontsize=12)
    ax.set_ylabel("Mean Rank IC", fontsize=12)
    ax.set_title("V7 Spatial — Monthly IC Trend (Jan-May 2026)\n(*** p<0.001, ns = not significant)", fontsize=13, fontweight="bold")
    ax.grid(axis="y", alpha=0.3, zorder=0)
    ax.set_ylim(ymin * 1.8, ymax * 1.25)

    # Regime switch annotation
    ax.axvspan(2.5, 4.5, alpha=0.08, color="red")
    ax.annotate("Regime Switch\n(IC→0)", (3.5, ymax * 0.7), ha="center", fontsize=10, color="red", alpha=0.7)
    save(fig, "fig17_monthly_ic_trend.png")


# ═══════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════
if __name__ == "__main__":
    print("Generating charts for experiment report...")
    print(f"   Output folder: {OUTDIR}\n")

    fig1_model_comparison()
    fig2_version_ic_evolution()
    fig3_strategy_comparison()
    fig4_window_comparison()
    fig5_knn_proxy_heatmap()
    fig6_spatial_v1_v2()
    fig7_k_sweep()
    fig8_cap_ic()
    fig9_nk_sweep_heatmap()
    fig10_monthly_decomposition()
    fig11_cumulative_returns()
    fig12_v8_vs_v7()
    fig13_loss_comparison()
    fig14_feature_ic_ranking()
    fig15_lambda_sweep()
    fig16_jan_decomposition()
    fig17_monthly_ic_trend()

    print(f"\nDone! 17 charts saved to {OUTDIR}/")
