#!/usr/bin/env python3
"""
plot_report.py — 实验报告图表生成（基于真实数据）

输入:
  - daily_scores.parquet (各模型打分)
  - all_data.parquet (OHLCV + 收益率)
  - benchmark_data.parquet (CSI300 基准)
  - Wu backtest_n5k3_daily.csv (统一回测)

输出: figures/ 目录下 PNG 图表

用法:
  python plot_report.py --start 20260105 --end 20260529
  python plot_report.py --competition  # 仅竞赛期图表
"""
import os, sys, argparse
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.patches import FancyBboxPatch
import warnings
warnings.filterwarnings("ignore")

# ── Paths ──
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(SCRIPT_DIR)
OUTDIR = os.path.join(SCRIPT_DIR, "figures")
os.makedirs(OUTDIR, exist_ok=True)

PARQUET_ALL   = os.path.join(ROOT, "processed", "all_data.parquet")
BENCHMARK     = os.path.join(ROOT, "v3", "results", "benchmark_data.parquet")
SCORE_PATHS = {
    "V6 Spatial (T+5)": os.path.join(ROOT, "v6", "results", "daily_scores.parquet"),
    "V7 GRU (T+1)":     os.path.join(ROOT, "v7", "results", "daily_scores_gru_t1.parquet"),
    "V7 Spatial (T+1)": os.path.join(ROOT, "v7", "results", "daily_scores_spatial_t1.parquet"),
}
MODEL_COLORS = {
    "V6 Spatial (T+5)": "#42A5F5",
    "V7 GRU (T+1)":     "#66BB6A",
    "V7 Spatial (T+1)": "#EF5350",
    "CNN-LSTM (Wu)":    "#AB47BC",
    "沪深300":           "#9E9E9E",
}
WU_DAILY  = os.path.join(ROOT, "..", "train_wu", "CNN-LSTM", "result", "backtest", "backtest_n5k3_daily.csv")

# ── Style ──
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["SimHei", "DejaVu Sans", "Arial"],
    "axes.unicode_minus": False,
    "figure.dpi": 150, "savefig.dpi": 150,
    "savefig.bbox": "tight", "figure.facecolor": "white",
})

COLORS = {
    "V6_Spatial": "#42A5F5", "V7_GRU": "#66BB6A",
    "V7_Spatial": "#EF5350", "Wu_CNN-LSTM": "#AB47BC",
    "CSI300": "#9E9E9E", "pos": "#4CAF50", "neg": "#F44336",
    "alpha": "#2196F3", "beta": "#FF9800",
}

# ── Helpers ──
def save(fig, name):
    path = os.path.join(OUTDIR, name)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  [OK] {name}")


def load_csi300():
    """Load CSI300 daily returns from benchmark_data.parquet."""
    df = pd.read_parquet(BENCHMARK)
    df["trade_date"] = df["date"].astype(str)
    return df[["trade_date", "idx_ret", "csi5d"]].copy()


def run_nk_backtest(scores_df, returns_df, csi5d_map, n_hold=5, k_rotate=3):
    """
    Proper N-K rotation backtest (T+1 execution).
    - Day 0 signal: buy top-N, return realized on next day
    - Day i>0: held stocks earn return; rotate at most K stocks
    - Strategy B: CSI5d < -1% → reduce position to 80%
    - Cost: sell 0.076% + buy 0.026% per stock, applied to rotated portion
    """
    dates = sorted(scores_df["trade_date"].unique())
    if len(dates) < 2:
        return pd.DataFrame()

    # Build date-indexed score lookup for speed
    score_lookup = {}
    for d in dates:
        day = scores_df[scores_df["trade_date"] == d].set_index("ts_code")["score"]
        score_lookup[d] = day

    # Build date-indexed return lookup
    ret_lookup = {}
    for d in dates:
        ret_lookup[d] = returns_df[returns_df["trade_date"] == d].set_index("ts_code")["pct_chg"] / 100.0

    held = set()      # currently held ts_codes
    daily_rows = []

    for i, signal_d in enumerate(dates[:-1]):  # last date has no next-day return
        next_d = dates[i + 1]
        day_scores = score_lookup.get(signal_d, pd.Series(dtype=float))
        day_rets = ret_lookup.get(next_d, pd.Series(dtype=float))

        if len(day_scores) < n_hold:
            continue

        # Strategy B: risk-off
        csi5d = csi5d_map.get(signal_d, 0)
        risk_off = csi5d < -1.0
        n_target = max(1, int(n_hold * 0.8)) if risk_off else n_hold

        if i == 0:
            # First day: buy top-N
            top_n = set(day_scores.nlargest(n_target).index)
            top_n &= set(day_rets.index)
            port_ret = day_rets[list(top_n)].mean() if top_n else 0.0
            cost = (0.00076 + 0.00026) * n_target / n_hold
            held = top_n
        else:
            # Portfolio return: held stocks' return
            held_valid = held & set(day_rets.index)
            port_ret = day_rets[list(held_valid)].mean() if held_valid else 0.0

            # Rotate: sell lowest-ranked in held, buy highest-ranked not held
            top_n = set(day_scores.nlargest(n_target).index) & set(day_rets.index)

            held_ranked = sorted(held_valid, key=lambda x: day_scores.get(x, -1e9))
            to_sell = set(held_ranked[:k_rotate]) if len(held_ranked) >= k_rotate else set(held_ranked)
            to_sell &= held_valid

            candidates = (top_n - held_valid)
            to_buy = set(sorted(candidates, key=lambda x: day_scores.get(x, -1e9), reverse=True)[:k_rotate])

            n_traded = max(len(to_sell), len(to_buy))
            cost = (0.00076 + 0.00026) * n_traded / n_hold

            held = (held_valid - to_sell) | to_buy

        port_ret_net = port_ret - cost

        daily_rows.append({
            "signal_date": signal_d,
            "port_ret": port_ret_net,
            "port_ret_gross": port_ret,
            "n_hold": n_target,
            "risk_off": risk_off,
        })

    return pd.DataFrame(daily_rows)


# ═══════════════════════════════════════════════════════════
#  FIG A: Multi-model cumulative equity curves + CSI300
# ═══════════════════════════════════════════════════════════
def fig_equity_curves(model_equity, csi300_df, wu_df, start, end):
    fig, ax = plt.subplots(figsize=(14, 6))

    for name, df in model_equity.items():
        if df is None or len(df) == 0:
            continue
        cum = (1 + df["port_ret"]).cumprod()
        ax.plot(range(len(cum)), cum.values, color=MODEL_COLORS.get(name, "#333"),
                linewidth=1.8, label=name, zorder=3)

    # Wu (CNN-LSTM)
    if wu_df is not None and len(wu_df) > 0:
        wu = wu_df[(wu_df["signal_date"] >= start) & (wu_df["signal_date"] <= end)].copy()
        if len(wu) > 0:
            wu_cum = (1 + wu["port_ret"].values / 100.0).cumprod()
            ax.plot(range(len(wu_cum)), wu_cum, color=MODEL_COLORS["CNN-LSTM (Wu)"],
                    linewidth=1.8, linestyle="--", label="CNN-LSTM (Wu)", zorder=3)

    # CSI300
    csi = csi300_df[(csi300_df["trade_date"] >= start) & (csi300_df["trade_date"] <= end)].copy()
    if len(csi) > 0:
        csi_cum = (1 + csi["idx_ret"].values / 100.0).cumprod()
        ax.plot(range(len(csi_cum)), csi_cum, color=MODEL_COLORS["沪深300"],
                linewidth=2, linestyle=":", label="沪深300", zorder=2)

    ax.axhline(y=1.0, color="black", linewidth=0.5, linestyle="--", alpha=0.3)
    ax.set_xlabel("交易日", fontsize=12)
    ax.set_ylabel("累计收益 (倍)", fontsize=12)
    ax.set_title(f"统一回测 — 累计收益曲线\n({start} → {end}, N=5 K=3, 含成本)",
                 fontsize=14, fontweight="bold")
    ax.legend(fontsize=10, loc="upper left")
    ax.grid(alpha=0.25)
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda y, _: f"{y:.2f}"))
    save(fig, "fig_equity_curves.png")


# ═══════════════════════════════════════════════════════════
#  FIG B: Competition period (Jun 1-12) daily + cumulative
# ═══════════════════════════════════════════════════════════
def fig_competition(comp_daily, csi300_df):
    if comp_daily is None or len(comp_daily) == 0:
        print("  [SKIP] fig_competition — 无数据")
        return

    df = comp_daily.copy()
    date_col = "return_date" if "return_date" in df.columns else "signal_date"
    df["cum"] = (1 + df["port_ret"]).cumprod()

    # CSI300 for same period
    csi_comp = None
    if csi300_df is not None:
        csi = csi300_df[csi300_df["trade_date"].between("20260601", "20260612")].copy()
        if len(csi) > 0:
            csi_comp = (1 + csi["idx_ret"].values / 100.0).cumprod()

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(13, 7.5), sharex=True,
                                    gridspec_kw={"height_ratios": [1.1, 1.3]})

    x = range(len(df))
    daily_pct = df["port_ret"].values * 100
    colors_bar = [COLORS["pos"] if r > 0 else COLORS["neg"] for r in daily_pct]

    # Top: daily return bars
    ax1.bar(x, daily_pct, color=colors_bar, edgecolor="white", width=0.55, zorder=3)
    ax1.axhline(y=0, color="black", linewidth=0.5)
    for i, (d, r) in enumerate(zip(df[date_col], daily_pct)):
        y = r + (0.35 if r >= 0 else -0.7)
        ax1.text(i, y, f"{r:+.1f}%", ha="center", fontsize=9,
                 fontweight="bold", color=COLORS["pos"] if r >= 0 else COLORS["neg"])
    ax1.set_ylabel("日收益率 (%)", fontsize=12)
    ax1.set_title("模拟交易竞赛期 — 每日收益 (2026年6月1–12日)", fontsize=14, fontweight="bold")
    ax1.grid(axis="y", alpha=0.2)

    # Bottom: cumulative line chart
    ax2.plot(x, df["cum"].values, "o-", color=MODEL_COLORS["V7 Spatial (T+1)"], linewidth=2.5, markersize=8,
             label="V7 Spatial 策略", zorder=3)
    if csi_comp is not None and len(csi_comp) >= len(df):
        ax2.plot(x, csi_comp[:len(df)], "s--", color=MODEL_COLORS["沪深300"], linewidth=2, markersize=6,
                 label="沪深300基准", zorder=2)

    ax2.axhline(y=1.0, color="black", linewidth=0.5, linestyle="--", alpha=0.3)
    for i, (d, c) in enumerate(zip(df[date_col], df["cum"])):
        ax2.annotate(f"{c:.4f}", (i, c), textcoords="offset points",
                     xytext=(0, 12), fontsize=8, ha="center", fontweight="bold")
    ax2.set_xticks(x)
    ax2.set_xticklabels([f"6/{d[4:6]}" for d in df[date_col]], fontsize=10)
    ax2.set_ylabel("累计收益 (倍)", fontsize=12)
    ax2.set_xlabel("日期", fontsize=12)
    ax2.grid(alpha=0.2)
    ax2.legend(fontsize=10, loc="upper left")

    final_ret = (df["cum"].iloc[-1] - 1) * 100
    fig.suptitle(f"竞赛最终收益: {final_ret:+.2f}%", fontsize=12, color="#555", y=1.01)
    save(fig, "fig_competition.png")


# ═══════════════════════════════════════════════════════════
#  FIG C: Daily IC timeseries + rolling
# ═══════════════════════════════════════════════════════════
def fig_daily_ic(ic_data):
    """ic_data: dict of {model_name: DataFrame with [trade_date, rank_ic]}"""
    fig, ax = plt.subplots(figsize=(14, 5.5))

    for name, df in ic_data.items():
        if df is None or len(df) == 0:
            continue
        df = df.sort_values("trade_date").copy()
        x = range(len(df))
        # Rolling 10-day
        roll = df["rank_ic"].rolling(10, min_periods=3).mean()
        ax.plot(x, roll.values, color=MODEL_COLORS.get(name, "#333"),
                linewidth=2, alpha=0.85, label=f"{name} (10d MA)")
        # Scatter daily
        ax.scatter(x, df["rank_ic"].values, color=MODEL_COLORS.get(name, "#333"),
                   alpha=0.15, s=12, zorder=2)

    ax.axhline(y=0, color="black", linewidth=0.5)
    ax.set_xlabel("交易日", fontsize=12)
    ax.set_ylabel("Rank IC", fontsize=12)
    ax.set_title("日度 Rank IC — 10日滚动均值", fontsize=14, fontweight="bold")
    ax.legend(fontsize=10, loc="upper right")
    ax.grid(alpha=0.25)
    save(fig, "fig_daily_ic.png")


# ═══════════════════════════════════════════════════════════
#  FIG D: Monthly IC boxplot
# ═══════════════════════════════════════════════════════════
def fig_monthly_ic_box(ic_data):
    fig, ax = plt.subplots(figsize=(12, 5.5))
    months = ["Jan", "Feb", "Mar", "Apr", "May"]
    positions = []
    labels = []
    all_data = []

    for mi, month in enumerate(months):
        for ni, (name, df) in enumerate(ic_data.items()):
            if df is None or len(df) == 0:
                continue
            df = df.copy()
            df["month"] = df["trade_date"].str[4:6]
            m_data = df[df["month"] == f"{mi+1:02d}"]["rank_ic"].dropna().values
            if len(m_data) > 0:
                pos = mi * (len(ic_data) + 1) + ni
                positions.append(pos)
                labels.append(name if mi == 0 else "")
                all_data.append({"pos": pos, "data": m_data, "color": MODEL_COLORS.get(name, "#333")})

    if not all_data:
        print("  [SKIP] fig_monthly_ic_box — no data")
        return

    bp = ax.boxplot([d["data"] for d in all_data], positions=[d["pos"] for d in all_data],
                    patch_artist=True, widths=0.6, showfliers=False,
                    medianprops={"color": "black", "linewidth": 1.5})

    for patch, d in zip(bp["boxes"], all_data):
        patch.set_facecolor(d["color"])
        patch.set_alpha(0.5)

    # Month dividers
    for mi in range(1, len(months)):
        ax.axvline(x=mi * (len(ic_data) + 1) - 0.5, color="gray", linestyle="--", alpha=0.3, linewidth=0.8)

    ax.axhline(y=0, color="black", linewidth=0.5)
    ax.set_xticks([mi * (len(ic_data) + 1) + (len(ic_data) - 1) / 2 for mi in range(len(months))])
    ax.set_xticklabels([f"{m}月" for m in months], fontsize=12)
    ax.set_ylabel("日度 Rank IC", fontsize=12)
    ax.set_title("月度 Rank IC 分布", fontsize=14, fontweight="bold")
    ax.grid(axis="y", alpha=0.25)

    # Legend (custom)
    from matplotlib.patches import Patch
    legend_patches = [Patch(facecolor=COLORS.get(n, "#333"), alpha=0.5, label=n)
                      for n in ic_data.keys()]
    ax.legend(handles=legend_patches, fontsize=9, loc="lower left")
    save(fig, "fig_monthly_ic_box.png")


# ═══════════════════════════════════════════════════════════
#  FIG E: IC histogram overlay
# ═══════════════════════════════════════════════════════════
def fig_ic_histogram(ic_data):
    fig, ax = plt.subplots(figsize=(10, 5.5))
    for name, df in ic_data.items():
        if df is None or len(df) == 0:
            continue
        vals = df["rank_ic"].dropna().values
        ax.hist(vals, bins=25, alpha=0.4, color=MODEL_COLORS.get(name, "#333"),
                label=f"{name} (μ={vals.mean():.3f})", density=True)

    ax.axvline(x=0, color="black", linewidth=0.5, linestyle="--")
    ax.set_xlabel("Rank IC", fontsize=12)
    ax.set_ylabel("密度", fontsize=12)
    ax.set_title("Rank IC 分布对比", fontsize=14, fontweight="bold")
    ax.legend(fontsize=10)
    ax.grid(axis="y", alpha=0.25)
    save(fig, "fig_ic_histogram.png")


# ═══════════════════════════════════════════════════════════
#  FIG F: Radar chart — multi-model metrics
# ═══════════════════════════════════════════════════════════
def fig_radar(metrics_dict):
    """
    Normalized grouped bar chart — each metric independently scaled to [0,1]
    across models for fair visual comparison.
    """
    if not metrics_dict:
        return
    models = list(metrics_dict.keys())
    metric_names = ["夏普比率", "IC", "ICIR", "胜率", "年化收益", "抗回撤"]
    metric_keys = ["sharpe", "ic", "icir", "win_rate", "annual_ret", "anti_dd"]

    # Build data matrix [n_models, n_metrics]
    data = np.zeros((len(models), len(metric_keys)))
    for mi, name in enumerate(models):
        m = metrics_dict[name]
        data[mi, 0] = m.get("sharpe", 0)
        data[mi, 1] = m.get("ic", 0)
        data[mi, 2] = m.get("icir", 0)
        data[mi, 3] = m.get("win_rate", 50)
        data[mi, 4] = m.get("annual_ret", 0)
        data[mi, 5] = 1 - abs(m.get("max_dd", 0))

    # Normalize each metric to [0, 1] across models
    data_norm = np.zeros_like(data)
    for j in range(len(metric_keys)):
        col = data[:, j]
        cmin, cmax = col.min(), col.max()
        if cmax - cmin < 1e-8:
            data_norm[:, j] = 0.5
        else:
            data_norm[:, j] = (col - cmin) / (cmax - cmin)

    fig, ax = plt.subplots(figsize=(12, 5.5))
    x = np.arange(len(metric_names))
    w = 0.22
    for mi, name in enumerate(models):
        offset = (mi - (len(models) - 1) / 2) * w
        bars = ax.bar(x + offset, data_norm[mi], w, color=MODEL_COLORS.get(name, "#333"),
                      alpha=0.85, label=name, zorder=3)
        # Annotate with actual values
        for j, bar in enumerate(bars):
            val = data[mi, j]
            fmt = f"{val:.2f}" if abs(val) < 10 else f"{val:.1f}"
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                    fmt, ha="center", fontsize=7, fontweight="bold", rotation=0)

    ax.set_xticks(x)
    ax.set_xticklabels(metric_names, fontsize=11)
    ax.set_ylabel("归一化得分", fontsize=11)
    ax.set_title("多维度模型对比（各指标独立归一化）", fontsize=14, fontweight="bold")
    ax.legend(fontsize=9, loc="upper right")
    ax.set_ylim(0, 1.25)
    ax.grid(axis="y", alpha=0.2)
    save(fig, "fig_radar.png")


# ═══════════════════════════════════════════════════════════
#  FIG G: Monthly return decomposition
# ═══════════════════════════════════════════════════════════
def fig_monthly_returns(model_equity, csi300_df):
    fig, ax = plt.subplots(figsize=(12, 5.5))
    months = ["Jan", "Feb", "Mar", "Apr", "May"]
    x = np.arange(len(months))
    w = 0.18

    for ni, (name, df) in enumerate(model_equity.items()):
        if df is None or len(df) == 0:
            continue
        df = df.copy()
        df["month"] = df["signal_date"].str[4:6]
        monthly = []
        for mi in range(5):
            m_data = df[df["month"] == f"{mi+1:02d}"]
            if len(m_data) > 0:
                cum = (1 + m_data["port_ret"]).prod() - 1
            else:
                cum = np.nan
            monthly.append(cum * 100)
        ax.bar(x + ni * w, monthly, w, color=MODEL_COLORS.get(name, "#333"),
               alpha=0.85, label=name, zorder=3)

    ax.axhline(y=0, color="black", linewidth=0.5)
    ax.set_xticks(x + w * 1.5)
    ax.set_xticklabels([f"{m}月" for m in months], fontsize=12)
    ax.set_ylabel("月收益 (%)", fontsize=12)
    ax.set_title("各模型月度收益分解", fontsize=14, fontweight="bold")
    ax.legend(fontsize=9)
    ax.grid(axis="y", alpha=0.25)
    save(fig, "fig_monthly_returns.png")


# ═══════════════════════════════════════════════════════════
#  MAIN
# ═══════════════════════════════════════════════════════════
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", default="20260105")
    parser.add_argument("--end", default="20260529")
    parser.add_argument("--competition", action="store_true")
    parser.add_argument("--n-hold", type=int, default=5)
    parser.add_argument("--k-rotate", type=int, default=3)
    args = parser.parse_args()

    START, END = args.start, args.end
    N_HOLD, K_ROTATE = args.n_hold, args.k_rotate

    print(f"plot_report.py — {START} → {END}  N={N_HOLD} K={K_ROTATE}")
    print(f"  Output: {OUTDIR}/\n")

    # ── 1. Load CSI300 ──
    csi300_df = None
    if os.path.exists(BENCHMARK):
        csi300_df = load_csi300()
        csi5d_map = dict(zip(csi300_df["trade_date"], csi300_df["csi5d"]))
        print(f"[data] CSI300: {len(csi300_df)} days")
    else:
        print("[WARN] benchmark_data.parquet not found — CSI300 unavailable")
        csi5d_map = {}

    # ── 2. Load returns data (with next-day shift for IC alignment) ──
    returns_df = None
    returns_next = None
    if os.path.exists(PARQUET_ALL):
        ret = pd.read_parquet(PARQUET_ALL)
        ret["trade_date"] = ret["trade_date"].astype(str)
        returns_df = ret[["trade_date", "ts_code", "pct_chg"]].copy()
        returns_df["pct_chg"] = pd.to_numeric(returns_df["pct_chg"], errors="coerce")
        print(f"[data] returns: {returns_df['trade_date'].nunique()} dates")

        # Build next-day return lookup: signal_date D → pct_chg on D+1
        all_dates = sorted(returns_df["trade_date"].unique())
        date_to_next = {all_dates[i]: all_dates[i+1] for i in range(len(all_dates)-1)}
        returns_next = returns_df.copy()
        returns_next["signal_date"] = returns_next["trade_date"].map(
            {v: k for k, v in date_to_next.items()}
        )
        returns_next = returns_next.dropna(subset=["signal_date"])
        returns_next = returns_next[["signal_date", "ts_code", "pct_chg"]].rename(
            columns={"signal_date": "trade_date"})
    else:
        print("[WARN] all_data.parquet not found — cannot compute equity curves")

    # ── 3. Load model scores & compute IC + equity ──
    ic_data = {}
    model_equity = {}

    for name, path in SCORE_PATHS.items():
        if not os.path.exists(path):
            print(f"[skip] {name} — file not found: {path}")
            ic_data[name] = None
            model_equity[name] = None
            continue

        scores = pd.read_parquet(path)
        scores["trade_date"] = scores["trade_date"].astype(str)
        scores = scores[scores["trade_date"].between(START, END)]
        print(f"[data] {name}: {len(scores)} rows, {scores['trade_date'].nunique()} dates")

        # Compute daily Rank IC: score(D) ~ pct_chg(D+1)
        if returns_next is not None:
            merged = scores.merge(
                returns_next, on=["trade_date", "ts_code"], how="inner", suffixes=("", "_ret")
            )
            if len(merged) > 0:
                daily_ic = merged.groupby("trade_date").apply(
                    lambda g: g["score"].rank().corr(g["pct_chg"].rank()) if len(g) >= 10 else np.nan
                ).reset_index(name="rank_ic")
                daily_ic = daily_ic.dropna(subset=["rank_ic"])
                ic_data[name] = daily_ic
                print(f"  IC: μ={daily_ic['rank_ic'].mean():.4f}  σ={daily_ic['rank_ic'].std():.4f}  "
                      f">0:{ (daily_ic['rank_ic']>0).mean()*100:.0f}%")
            else:
                ic_data[name] = None
        else:
            ic_data[name] = None

        # Compute N-K backtest equity
        if returns_next is not None:
            eq = run_nk_backtest(scores, returns_next, csi5d_map, N_HOLD, K_ROTATE)
            eq = eq[eq["signal_date"].between(START, END)]
            model_equity[name] = eq
            if len(eq) > 0:
                cum = (1 + eq["port_ret"]).prod()
                print(f"  Equity: cumulative={cum:.4f}×  ({ (cum-1)*100:+.2f}%)")
        else:
            model_equity[name] = None

    # ── 4. Load Wu data ──
    wu_df = None
    if os.path.exists(WU_DAILY):
        wu_df = pd.read_csv(WU_DAILY)
        wu_df["signal_date"] = wu_df["signal_date"].astype(str)
        print(f"[data] Wu: {len(wu_df)} days")
    else:
        print(f"[WARN] Wu backtest not found: {WU_DAILY}")

    # ── 5. Generate charts ──
    print("\nGenerating charts...")

    # A: Equity curves
    if any(v is not None for v in model_equity.values()):
        fig_equity_curves(model_equity, csi300_df, wu_df, START, END)

    # C: Daily IC
    if any(v is not None for v in ic_data.values()):
        fig_daily_ic(ic_data)

    # D: Monthly IC box
    if any(v is not None for v in ic_data.values()):
        fig_monthly_ic_box(ic_data)

    # E: IC histogram
    if any(v is not None for v in ic_data.values()):
        fig_ic_histogram(ic_data)

    # G: Monthly returns
    if any(v is not None for v in model_equity.values()):
        fig_monthly_returns(model_equity, csi300_df)

    # ── 6. Competition period ──
    if args.competition:
        COMP_START, COMP_END = "20260601", "20260612"
        COMP_SCORE_START = "20260529"
        print(f"\n[competition] scoring {COMP_SCORE_START} → {COMP_END}")
        for name, path in SCORE_PATHS.items():
            if not os.path.exists(path):
                continue
            scores = pd.read_parquet(path)
            scores["trade_date"] = scores["trade_date"].astype(str)
            scores = scores[scores["trade_date"].between(COMP_SCORE_START, COMP_END)]
            if len(scores) == 0:
                continue
            if returns_next is not None:
                comp_eq = run_nk_backtest(scores, returns_next, csi5d_map, N_HOLD, K_ROTATE)
                if len(comp_eq) == 0:
                    continue
                # Map signal_date → actual return date
                all_comp_dates = sorted(scores["trade_date"].unique())
                d2n = {all_comp_dates[i]: all_comp_dates[i+1] for i in range(len(all_comp_dates)-1)}
                comp_eq["return_date"] = comp_eq["signal_date"].map(d2n)
                comp_eq = comp_eq[comp_eq["return_date"].between(COMP_START, COMP_END)]
                if len(comp_eq) > 0:
                    fig_competition(comp_eq, csi300_df)
                    break

    # ── 7. Radar chart (if metrics available) ──
    radar_metrics = {}
    for name in model_equity:
        eq = model_equity.get(name)
        ic = ic_data.get(name)
        if eq is not None and len(eq) > 0 and ic is not None and len(ic) > 0:
            cum = (1 + eq["port_ret"]).prod()
            ann_ret = cum ** (252 / len(eq)) - 1
            daily_rets = eq["port_ret"].values
            sharpe = daily_rets.mean() / (daily_rets.std() + 1e-8) * np.sqrt(252)
            peak = np.maximum.accumulate((1 + daily_rets).cumprod())
            max_dd = ((1 + daily_rets).cumprod() / peak - 1).min()
            radar_metrics[name] = {
                "sharpe": round(sharpe, 2),
                "ic": round(ic["rank_ic"].mean(), 3),
                "icir": round(ic["rank_ic"].mean() / (ic["rank_ic"].std() + 1e-8), 2),
                "win_rate": round((eq["port_ret"] > 0).mean() * 100, 1),
                "annual_ret": round(ann_ret * 100, 1),
                "max_dd": round(max_dd, 3),
            }

    if radar_metrics:
        fig_radar(radar_metrics)

    print(f"\nDone! Charts saved to {OUTDIR}/")


if __name__ == "__main__":
    main()
