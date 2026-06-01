"""Why cumulative return < CSI300 despite beating on many windows? Variance drag analysis."""
import numpy as np, pandas as pd, os
d = os.path.dirname(os.path.abspath(__file__))

rec = pd.read_parquet(os.path.join(d, "results", "benchmark_data.parquet"))
t20 = rec["top20_mean"].values
idx = rec["idx_ret"].values
csi5d = rec["csi5d"].values
dates = rec["date"].values
n = len(t20)

# ========== 1. Arithmetic vs Geometric (variance drag) ==========
arith_model = t20.mean()
arith_idx   = idx.mean()
geo_model   = np.exp(np.mean(np.log(1 + t20 / 100))) - 1
geo_idx     = np.exp(np.mean(np.log(1 + idx   / 100))) - 1
drag_model  = arith_model / 100 - geo_model
drag_idx    = arith_idx   / 100 - geo_idx

print("=" * 65)
print("SECTION 1: Why arithmetic wins don't become geometric wins")
print("=" * 65)
print(f"  Arithmetic daily mean:  Model={arith_model:+.3f}%  CSI300={arith_idx:+.3f}%")
print(f"  Geometric daily mean:   Model={geo_model*100:+.3f}%  CSI300={geo_idx*100:+.3f}%")
print(f"  Daily std:              Model={t20.std():.3f}%  CSI300={idx.std():.3f}%")
print(f"  Variance drag:          Model={drag_model*100:.3f}%  CSI300={drag_idx*100:.3f}%")
print(f"  → Model std is {t20.std()/idx.std():.1f}x CSI300, variance drag is {drag_model/drag_idx:.1f}x bigger")
print(f"  → Variance drag alone costs {(drag_model-drag_idx)*100:.2f}% of daily return")

# ========== 2. Asymmetric loss magnitude ==========
wins   = t20 > idx
losses = t20 < idx
ties   = t20 == idx

print(f"\n{'='*65}")
print("SECTION 2: Win-rate doesn't matter if losses are bigger")
print("="*65)
print(f"  Beat CSI300: {wins.sum()}/{n} days ({wins.sum()/n*100:.0f}%)")
print(f"  Lose to CSI:  {losses.sum()}/{n} days ({losses.sum()/n*100:.0f}%)")
print(f"  When WIN:  model={t20[wins].mean():+.3f}%  csi300={idx[wins].mean():+.3f}%  excess={t20[wins].mean()-idx[wins].mean():+.3f}%")
print(f"  When LOSE: model={t20[losses].mean():+.3f}%  csi300={idx[losses].mean():+.3f}%  excess={t20[losses].mean()-idx[losses].mean():+.3f}%")
print(f"  Loss magnitude ratio: {abs(t20[losses].mean()-idx[losses].mean())/abs(t20[wins].mean()-idx[wins].mean()):.1f}x")

# ========== 3. The compounding killer: worst days ==========
print(f"\n{'='*65}")
print("SECTION 3: Compounding disaster — few bad days kill months")
print("="*65)

# Find the 10 worst model days
worst_idx = np.argsort(t20)[:10]
print(f"  TOP 10 worst model days:")
cum_impact = 1.0; cum_impact_idx = 1.0
for rank, wi in enumerate(worst_idx):
    imp = 1 + t20[wi] / 100
    imp_idx = 1 + idx[wi] / 100
    saved_val = t20[wi] * 0.8
    msg = "★ TRIGGERED" if csi5d[wi] < -1.0 else "missed"
    print(f"  #{rank+1:>2} {dates[wi]}  model={t20[wi]:+6.2f}%  CSI300={idx[wi]:+6.2f}%  "
          f"csi5d={csi5d[wi]:+5.1f}%  [{msg}]  "
          f"base if saved: {saved_val:+6.2f}%")
    cum_impact *= imp
    cum_impact_idx *= imp_idx

# What if strategy B was active on worst days?
strat_10worst = np.prod(1 + (t20[worst_idx] * np.where(csi5d[worst_idx] < -1.0, 0.8, 1.0)) / 100)
print(f"  Worst-10 cumulative impact:         model={cum_impact:.4f}  csi300={cum_impact_idx:.4f}")
print(f"  With strategy B on these 10 days:   {strat_10worst:.4f}")

# ========== 4. Counterfactual: remove worst 3 days ==========
exclude_3 = np.ones(n, dtype=bool)
exclude_3[worst_idx[:3]] = False
print(f"\n  Counterfactual — remove 3 worst days:")
print(f"    Without {dates[worst_idx[0]]}: model={np.prod(1+np.delete(t20,worst_idx[0])/100):.4f}  → equal to CSI300")
print(f"    Without {dates[worst_idx[0]]} & {dates[worst_idx[1]]} & {dates[worst_idx[2]]}: "
      f"model={np.prod(1+t20[exclude_3]/100):.4f}  vs CSI300={np.prod(1+idx[exclude_3]/100):.4f}")

# ========== 5. Daily compounding walk ==========
print(f"\n{'='*65}")
print("SECTION 4: The compounding math (simplified)")
print("="*65)
print("  If model beats CSI300 by 0.1% on 60 of 74 days,")
print("  but loses to CSI300 by 3% on 3 days:")
daily = np.array([0.1*60 + (-3.0)*3 + 0.0*(74-63)]) / 74  # roughly
geo = np.exp(np.log(1+0.001)*60/74 + np.log(1-0.03)*3/74) - 1
print(f"    Arithmetic mean daily:   +{0.1*60/74 - 3.0*3/74:+.3f}%")
print(f"    Geometric mean daily:    {geo*100:+.3f}%")
print(f"    Net effect: 60 small wins evaporated by 3 big losses")

# ========== 6. Strategy B: how much of the gap does it close? ==========
print(f"\n{'='*65}")
print("SECTION 5: Strategy B closes the gap")
print("="*65)
w = np.where(csi5d < -1.0, 0.8, 1.0)
sd = t20 * w
s_cum = np.prod(1 + sd / 100)
b_cum = np.prod(1 + t20 / 100)
i_cum = np.prod(1 + idx / 100)

print(f"  Baseline:   {b_cum:.4f}  (+{b_cum-1:+.2%})")
print(f"  CSI300:     {i_cum:.4f}  (+{i_cum-1:+.2%})")
print(f"  Strategy B: {s_cum:.4f}  (+{s_cum-1:+.2%})")
print(f"  Gap closed: {(s_cum-b_cum)/(i_cum-b_cum)*100:.0f}% of excess return gap")
print(f"  Strategy B > CSI300: {s_cum > i_cum}")

# ========== 7. The real story: model has higher beta ==========
beta_model_top20 = np.corrcoef(t20, idx)[0,1] * t20.std() / idx.std()
print(f"\n  Model top20 beta vs CSI300: {beta_model_top20:.2f}")
print(f"  → Model is a {beta_model_top20:.1f}x leveraged bet on CSI300")
print(f"  → When CSI300 drops -3%, model drops -3%*{beta_model_top20:.1f} = {-3*beta_model_top20:+.1f}%")
print(f"  → Compounding: (1-0.03)^{10} vs (1-0.03*{beta_model_top20:.1f})^{10}")
print(f"     = {(1-0.03)**10:.4f} vs {(1-0.03*beta_model_top20)**10:.4f}")
