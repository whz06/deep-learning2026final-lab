#!/usr/bin/env python3
"""
生成 fig_model_compare_bars.png — 剔除Wu的不可靠回测指标

布局: 2行 × 3列
  Sharpe / Rank IC / ICIR
  IC胜率 / 年化收益 / 最大回撤

Wu只参与IC/ICIR/IC胜率三个子图，其余三个子图仅画Li的三个模型。
对于Wu不参与的子图，bar位置不画柱、只标"N/A"。
"""
import os, numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO = r"D:\whz\course\深度学习\LAB\FINAL_LAB\deep-learning2026final-lab"
LI_REPORT = os.path.join(REPO, "train_li", "report")
OUTDIR = os.path.join(LI_REPORT, "figures")
os.makedirs(OUTDIR, exist_ok=True)

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["SimHei", "DejaVu Sans", "Arial"],
    "axes.unicode_minus": False,
    "figure.dpi": 150, "savefig.dpi": 150,
})

# ── 数据 ──
ic_df = pd.read_csv(os.path.join(LI_REPORT, "comparison", "ic_summary.csv"))
bt_df = pd.read_csv(os.path.join(LI_REPORT, "comparison", "backtest_summary.csv"))

ALL_MODELS = ["V6 Spatial\n(T+5)", "V7 GRU\n(T+1)", "V7 Spatial\n(T+1)", "CNN-LSTM\n(Wu)"]
ALL_COLORS = ["#42A5F5", "#66BB6A", "#EF5350", "#AB47BC"]

# Li models only
LI_MODELS = ALL_MODELS[:3]
LI_COLORS = ALL_COLORS[:3]

def get_li_data():
    """读取Li的三个模型数据"""
    pairs = [
        ("V6 Spatial\n(T+5)", "V6_Spatial", "V6_Spatial_N5"),
        ("V7 GRU\n(T+1)", "V7_GRU", "V7_GRU_N5"),
        ("V7 Spatial\n(T+1)", "V7_Spatial", "V7_Spatial_N5"),
    ]
    result = {}
    for disp, ic_key, bt_key in pairs:
        ic_row = ic_df[ic_df["model"].str.startswith(ic_key)].iloc[0]
        bt_row = bt_df[bt_df["model"].str.startswith(bt_key)].iloc[0]
        result[disp] = {
            "ic":         ic_row["mean_rank_ic"],
            "icir":       ic_row["icir"],
            "ic_win":     ic_row["ic_positive_pct"] / 100.0,
            "sharpe":     bt_row["sharpe"],
            "annual_ret": bt_row["annual_return_pct"] / 100.0,
            "max_dd":     bt_row["max_drawdown_pct"] / 100.0,
        }
    return result

li_data = get_li_data()

# Wu数据
wu_data = {"ic": 0.0667, "icir": 0.0667/0.1098, "ic_win": 0.72}

# 指标定义: (显示名, 键名, 格式化, 参与模型列表, 颜色列表)
METRICS_CONFIG = [
    ("Sharpe",   "sharpe",     "raw", LI_MODELS, LI_COLORS),
    ("Rank IC",  "ic",         "raw", ALL_MODELS, ALL_COLORS),
    ("ICIR",     "icir",       "raw", ALL_MODELS, ALL_COLORS),
    ("IC胜率",   "ic_win",     "pct", ALL_MODELS, ALL_COLORS),
    ("年化收益", "annual_ret", "pct", LI_MODELS, LI_COLORS),
    ("最大回撤", "max_dd",     "neg_pct", LI_MODELS, LI_COLORS),
]

fig, axes = plt.subplots(2, 3, figsize=(18, 10))
axes = axes.flatten()

for ai, (mname, mkey, fmt, models, colors) in enumerate(METRICS_CONFIG):
    ax = axes[ai]

    # 收集值
    vals = []
    for m in models:
        if m in li_data:
            v = li_data[m][mkey]
        else:
            v = wu_data[mkey]
        vals.append(v)

    # 转换为显示值
    display = []
    for v in vals:
        if fmt == "pct" or fmt == "neg_pct":
            display.append(v * 100.0)
        else:
            display.append(v)

    x_pos = np.arange(len(models))
    bar_width = 0.55

    # 画柱
    bars = ax.bar(x_pos, display, bar_width, color=colors, alpha=0.85,
                  edgecolor="white", linewidth=0.5, zorder=3)

    # 标注数值（固定在柱顶上方或柱底下方）
    for j, (bar, val) in enumerate(zip(bars, display)):
        if fmt == "neg_pct":
            label = f"{val:.1f}%"
        elif fmt == "pct":
            label = f"{val:.1f}%"
        else:
            label = f"{val:.2f}"

        if val >= 0:
            offset = max(display) * 0.04 if len(display) > 0 else 0.02
            y_pos = bar.get_height() + offset
            va = "bottom"
        else:
            offset = abs(min(display)) * 0.04 if len(display) > 0 else 0.02
            y_pos = bar.get_height() - offset
            va = "top"

        ax.text(bar.get_x() + bar.get_width() / 2, y_pos, label,
                ha="center", va=va, fontsize=9, fontweight="bold", color=colors[j])

    # 如果该子图有不参与的模型（Wu不在LI_MODELS中），在对应x位置画N/A标记
    all_x = np.arange(len(ALL_MODELS))
    participating = set(range(len(models)))
    for j in range(len(ALL_MODELS)):
        if j not in participating:
            # 画一个浅灰色虚线框占位
            ax.bar(j, 0, 0.55, color="none", edgecolor="#CCCCCC",
                   linewidth=1.5, linestyle="--", zorder=2)
            # 标N/A
            ax.text(j, 0, "N/A", ha="center", va="center", fontsize=10,
                    color="#999999", fontstyle="italic", fontweight="bold",
                    transform=ax.get_xaxis_transform())

    ax.set_xticks(all_x)
    ax.set_xticklabels(ALL_MODELS, fontsize=9)
    ax.set_title(mname, fontsize=13, fontweight="bold", pad=10)
    ax.axhline(y=0, color="black", linewidth=0.5)
    ax.grid(axis="y", alpha=0.2)

    # 自动调整y轴留出标注空间
    ymin, ymax = ax.get_ylim()
    if ymax > 0:
        ax.set_ylim(ymin, ymax * 1.15)
    if ymin < 0:
        ax.set_ylim(ymin * 1.25, ymax)

fig.suptitle("模型关键指标对比（Wu的Sharpe/年化收益/最大回撤因范式差异不适用）",
             fontsize=14, fontweight="bold", y=1.02)

plt.tight_layout()
out_path = os.path.join(OUTDIR, "fig_model_compare_bars.png")
fig.savefig(out_path, dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"[OK] {out_path}")
