#!/usr/bin/env python3
"""生成 fig_model_compare_bars.png — 只替换 Wu 的数据，不动 Li 的模型数据"""
import os, sys, json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

ROOT = os.path.dirname(os.path.abspath(__file__))
OUTDIR = os.path.join(ROOT, "figures")
os.makedirs(OUTDIR, exist_ok=True)

plt.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["SimHei", "DejaVu Sans", "Arial"],
                      "axes.unicode_minus": False, "figure.dpi": 150, "savefig.dpi": 150})

# ═══ Li 的模型数据（从 comparison CSV 读取，完全不动）═══
ic_csv = os.path.join(ROOT, "comparison", "ic_summary.csv")
bt_csv = os.path.join(ROOT, "comparison", "backtest_summary.csv")
ic_df = pd.read_csv(ic_csv)
bt_df = pd.read_csv(bt_csv)

# Li 的三个模型（只取 N=5 的回测数据）
li_models = {
    "V6 Spatial\n(T+5)": {"ic_key": "V6_Spatial", "bt_key": "V6_Spatial_N5"},
    "V7 GRU\n(T+1)":    {"ic_key": "V7_GRU",     "bt_key": "V7_GRU_N5"},
    "V7 Spatial\n(T+1)": {"ic_key": "V7_Spatial", "bt_key": "V7_Spatial_N5"},
}

# ═══ Wu 的数据（从自己回测结果计算）═══
wu_csv = r"D:\whz\course\深度学习\LAB\FINAL_LAB\deep-learning2026final-lab\train_wu\CNN-LSTM\result\backtest\backtest_n5k3_daily.csv"
wu_df = pd.read_csv(wu_csv)
wu_rets = wu_df["port_ret"].values / 100.0  # 转小数
wu_cum = float(np.cumprod(1 + wu_rets)[-1])
wu_n = len(wu_rets)
wu_ann = wu_cum ** (252 / wu_n) - 1
wu_vol = float(np.std(wu_rets) * np.sqrt(252))
wu_sharpe = (float(np.mean(wu_rets)) * 252) / wu_vol if wu_vol > 0 else 0.0
peak = np.maximum.accumulate(np.cumprod(1 + wu_rets))
wu_mdd = float((np.cumprod(1 + wu_rets) / peak - 1).min())

# RankIC from test_correct_backtest.py output
wu_rankic = 0.0667
wu_rankic_std = 0.1098
wu_icir = wu_rankic / wu_rankic_std if wu_rankic_std > 0 else 0
wu_win = 0.72  # RankIC > 0 比例（与 Li 模型口径一致）

# ═══ 组装绘图数据 ═══
metrics_def = [
    ("Sharpe", "sharpe", "bar"),
    ("RankIC", "ic", "bar"),
    ("ICIR", "icir", "bar"),
    ("胜率", "win_rate", "pct"),
    ("年化收益", "annual_ret", "pct"),
    ("最大回撤", "max_dd", "neg_pct"),
]

models_order = ["V6 Spatial\n(T+5)", "V7 GRU\n(T+1)", "V7 Spatial\n(T+1)", "CNN-LSTM\n(Wu)"]
colors = ["#42A5F5", "#66BB6A", "#EF5350", "#AB47BC"]

data = {}
for m in models_order:
    data[m] = {}

for m_display, m_key in [
    ("V6 Spatial\n(T+5)", "V6_Spatial"), ("V7 GRU\n(T+1)", "V7_GRU"), ("V7 Spatial\n(T+1)", "V7_Spatial")]:
    ic_row = ic_df[ic_df["model"].str.startswith(m_key)].iloc[0]
    bt_row = bt_df[bt_df["model"].str.startswith(m_key + "_N5")].iloc[0]
    data[m_display] = {
        "sharpe": bt_row["sharpe"],
        "ic": ic_row["mean_rank_ic"],
        "icir": ic_row["icir"],
        "win_rate": ic_row["ic_positive_pct"] / 100,
        "annual_ret": bt_row["annual_return_pct"] / 100,
        "max_dd": bt_row["max_drawdown_pct"] / 100,
    }

# Wu 的数据
data["CNN-LSTM\n(Wu)"] = {
    "sharpe": wu_sharpe,
    "ic": wu_rankic,
    "icir": wu_icir,
    "win_rate": wu_win,
    "annual_ret": wu_ann,
    "max_dd": wu_mdd,
}

# ═══ 绘图 ═══
fig, axes = plt.subplots(2, 3, figsize=(18, 10))
axes = axes.flatten()
x = np.arange(len(models_order))
w = 0.55

for ai, (mname, mkey, fmt) in enumerate(metrics_def):
    ax = axes[ai]
    values = [data[m][mkey] for m in models_order]

    if fmt == "pct":
        display_vals = [v * 100 for v in values]
        ylabel = "%"
    elif fmt == "neg_pct":
        display_vals = [v * 100 for v in values]
        ylabel = "%"
    else:
        display_vals = values
        ylabel = ""

    bars = ax.bar(x, display_vals, w, color=colors, alpha=0.85, edgecolor="white", linewidth=0.5, zorder=3)

    # 动态计算标注偏移（基于数据范围的百分比）
    y_range = max(display_vals) - min(display_vals) if max(display_vals) != min(display_vals) else 1.0
    offset = y_range * 0.03  # 数据范围的 3%
    # 标注数值
    for j, (bar, val) in enumerate(zip(bars, display_vals)):
        if fmt == "neg_pct":
            label = f"{val:.1f}%"
        elif fmt == "pct":
            label = f"{val:.1f}%"
        else:
            label = f"{val:.2f}"
        y_pos = bar.get_height() + (offset if val >= 0 else -offset * 2)
        ax.text(bar.get_x() + bar.get_width()/2, y_pos, label,
                ha="center", fontsize=9, fontweight="bold", color=colors[j])

    ax.set_xticks(x)
    ax.set_xticklabels(models_order, fontsize=9)
    ax.set_title(mname, fontsize=13, fontweight="bold", pad=10)
    ax.axhline(y=0, color="black", linewidth=0.5)
    ax.grid(axis="y", alpha=0.2)

    if ai == 0:
        ax.set_ylabel(ylabel, fontsize=10)

fig.suptitle("四模型关键指标对比（统一回测框架，N=5 K=3）", fontsize=16, fontweight="bold", y=1.02)

plt.tight_layout()
out_path = os.path.join(OUTDIR, "fig_model_compare_bars.png")
fig.savefig(out_path, dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"[OK] {out_path}")
