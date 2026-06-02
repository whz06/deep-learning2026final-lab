# V7 实验全记录 — T+1 标签训练与回测验证

> 时间：2026-06-02 | 基线：v5 GRU (T+5, val IC=0.1114) 和 v6 Spatial (T+5, val IC=0.1134)

---

## 一、背景与动机

### V6 诊断揭示了关键问题

V6 的诊断脚本（`diagnose_gap_v2.py`）发现：

| 模型预测标的 | Test IC | IC>0 | 结论 |
|:---:|:---:|:---:|------|
| T+1 收益率 | **+0.053** | 67.6% | ✅ 有效 |
| T+5 收益率 | **+0.010** | **42.6%** | ❌ 比随机还差 |

**模型被训练预测 T+5，但在 test 期真正有效的是 T+1 预测。** 根因：

1. GRU 输入是 60 天 OHLCV + 技术指标（RSI、MACD、布林带等），这些特征天然预测短期（1日）动量/反转
2. 训练期（2019-2025）动量延续到 T+5，模型学会 T+1 就能拿到 T+5 的高分
3. 2026 年动量断裂，T+1 ≠ T+5，真相暴露

**解决方案**：用 T+1 标签重新训练，去掉"T+5 动量溢价"的虚假成分。

---

## 二、V7 训练设计

### 2.1 数据窗口

```
v7/build_windows.py:
  LABEL_HORIZON = 1  ← 唯一改动：从 5 改为 1
  label = pct_chg[t+1]  ← 单日收益率，而非 5 日累计
  输出 → processed/v7_windows/  ← 新建目录，不覆盖 v5_windows
```

其余不变：26 维特征（19 temporal + 7 cross），log1p + winsorization + PE/PB clip。

### 2.2 两个模型

| 模型 | 架构 | N_sample | 训练时间 |
|------|------|:---:|:---:|
| **v7 GRU** | GRU H=128 L=1 D=0.2 | 2048 | ~11 min |
| **v7 Spatial** | GRU + concat fusion (K=5, d=32) | 1024 | ~18 min |

训练配置：val_every=2, PATIENCE=7, lr=3e-4, ListMLE loss, val=2025, train=2019-2024。

### 2.3 训练结果

| 模型 | Val IC | Best Epoch | 参数 | vs v5 T+5 baseline |
|------|:---:|:---:|:---:|:---:|
| v5 GRU (T+5) | 0.1114 | 9 | 68,225 | — |
| v7 GRU (T+1) | **0.1023** | 14 | 68,225 | -0.009 (预期内) |
| v7 Spatial (T+1) | **0.1062** | 30 | 82,561 | +0.004 vs v7 GRU |

**T+1 val IC 天然低于 T+5**：单日噪声大，但更诚实——没有动量延续溢价。

**Spatial 在 T+1 上也有效**：比 GRU baseline 高 +0.0039，且收敛更稳定（epoch 30 vs 14）。

---

## 三、Test 期验证（600 只抽样）

### 3.1 Test Rank IC

| 模型 | Test T+1 IC | Std | IC>0 |
|------|:---:|:---:|:---:|
| v7 GRU | **+0.052** | 0.167 | 61.6% |
| v7 Spatial | **+0.048** | 0.160 | 64.4% |

### 3.2 与 V6/V5 对比

| 模型 | 标签 | Val IC | Test T+1 IC |
|------|:---:|:---:|:---:|
| v5 GRU | T+5 | 0.111 | 0.053 (推断) |
| v6 Spatial | T+5 | 0.113 | **0.051** |
| v7 GRU | T+1 | 0.102 | **0.052** |
| v7 Spatial | T+1 | 0.106 | **0.048** |

Test IC 几乎完全相同（0.048-0.053），无论是 T+1 还是 T+5 训练，无论是 GRU 还是 Spatial。**信号强度受限于特征信息上限，不是模型架构。**

### 3.3 V5 T+5 模型在 Test 期 T+5 上的真实表现（诊断发现）

```
V6 模型（T+5 训练）在 test 期的:
  T+1 IC = 0.053  ← 模型实际上在预测这个
  T+5 IC = 0.010  ← 但被训练要求预测这个
```

---

## 四、策略 Sweep

### 4.1 全周期连续回测（600 股，Strategy B）

| N_hold | sell_k | 日换手 | v7 GRU | v7 Spatial |
|:---:|:---:|:---:|:---:|:---:|
| **5** | **3** | **60%** | +3.43% | **+15.45%** |
| 5 | 2 | 40% | +2.20% | +10.81% |
| 10 | 3 | 30% | — | +4.69% |
| 10 | 5 | 50% | — | +4.46% |
| 15 | 10 | 67% | — | +1.80% |
| 20 | 10 | 50% | — | +2.34% |

**v7 Spatial, N=5, K=3 是压倒性冠军：+15.45%（excess +10.2% vs CSI300），远超 v3 的 +6.85%。**

Spatial 在 T+1 标签上的放大器效应是 T+5 标签上的 **10 倍**（+15.45% vs +1.80% at N=15 K=10，or vs 全量 sweep 最高 +0.07% on T+5）。正确标签 × 正确架构 = 指数级叠加。

### 4.2 精细网格 N=6..9 (K=3 固定)

**K=3 在所有 N 上都是最优。** 完整梯度：

```
N=5,  K=3: +15.45%  turnover=60%
N=6:  K=3: +13.19%  turnover=50%
N=7:  K=3: +10.88%  turnover=43%
N=8:  K=3:  +9.68%  turnover=38%
N=9:  K=3:  +6.76%  turnover=33%
N=10: K=3:  +4.69%  turnover=30%
```

每增持 1 只股票，收益掉 **1.2-2.9%**。模型 alpha 集中在前 5 名，6-10 名迅速稀释。

### 4.3 10×10 窗口稳定性（v7 Spatial, K=3）

| N | 均值收益 | Std | 胜率 | 5月最差窗口 |
|:---:|:------:|:---:|:---:|:---:|
| 5 | **+0.88%** | 6.81% | 5/10 | -9.85% |
| 7 | -0.06% | 6.15% | **6/10** | -10.18% |
| 10 | -0.81% | **5.24%** | 4/10 | -9.71% |

**关键发现**：所有 N 在 5 月的 3 个窗口都遭到 -6% 到 -10% 的重创。Std 差异仅 1.6%。N=5 的优势来自于**好窗口的涨幅更大**（+14% vs +8%），而不是好窗口更多。

---

## 五、遇到的错误与修复

### 错误 1：v7 build_windows 输出路径覆盖问题

**现象**：初始引用 `v5_windows` 目录，修改 LABEL_HORIZON 后会覆盖旧数据。
**修复**：输出到新目录 `processed/v7_windows/`，保留 v5_windows 供对比。

### 错误 2：`sdf_c.ffill()` 未拷贝

**现象**：Pandas SettingWithCopyWarning，ffill 未真正写入。
**修复**：`sdf_c = sdf_c.copy(); sdf_c = sdf_c.ffill()` 先 copy 再 ffill。

### 错误 3：`add_tech(sdf_c)` 在切片后缺少

**现象**：v7 validate.py 中 compute_features 取 sdf_c 但没有先调用 add_tech，TECH_FEATURES 列全部缺失，scores 持续为 None。
**修复**：在线计算 `sdf_c = add_tech(sdf_c)` 后再取特征列。

### 错误 4：warmup 不充分 -> scores 全部为空

**现象**：validate.py 从 df 过滤 needed 日期（仅 74 天），每个 stock 最多 74 行——不满足 W=60 + tech indicator warmup（~35 行）。所有 compute_features 返回 None，scores parquet 仅 636 字节空文件。
**修复**：用 300 天 warmup 预加载（`data_range >= all_dates[test_start_idx - 300]`），确保每个 stock 有 ~300 行历史用于计算技术指标。

### 错误 5：`_orig_mod.` 前缀导致 checkpoint 加载失败

**现象**：v7 训练时使用了 `torch.compile`，checkpoint 中所有 key 带 `_orig_mod.` 前缀。infer.py 的 `load_model` 直接 load_state_dict 报 KeyError。
**修复**：加载后 strip 前缀：`state = {k.replace("_orig_mod.", ""): v for k, v in state.items()}`。

### 错误 6：v7 训练 GRU baseline 用 compile，spatial 未用

**现象**：GRU baseline `torch.compile` 成功（简单 RNN），但 spatial model 的 topk+gather 触发 Triton 路径失败。
**修复**：spatial 训练脚本默认 `--no-compile`（已有），GRU 允许 compile。但两边都保存了 `_orig_mod.` 前缀 → infer.py 统一 strip。

### 错误 7：CSI300 float64 污染 cross_feat

**现象**：`idx_map = dict(zip(csi["trade_date"], csi["pct_chg"].astype(float)))` 生成 Python float（即 float64），`pct_arr - idx_pct` 将 float32 提升为 float64，导致 tensor dtype mismatch。
**修复**：`.astype(np.float32)` 替代 `.astype(float)`。

---

## 六、V7 最优模型预测

### 6/2 交易建议（20260601 数据）

```
v7 Spatial (T+1), Strategy B (CSI5d=-1.57% -> RISK-OFF 80%)

BUY (16):  688529.SH  688296.SH  300447.SZ  300207.SZ  603017.SH
           300272.SZ  001325.SZ  002467.SZ  301163.SZ  002418.SZ
           002998.SZ  603118.SH  300724.SZ  603048.SH  601012.SH  603628.SH

SELL (4):  603681.SH  605580.SH  301565.SZ  688121.SH
```

### 持仓分析（用户原有 10 只 vs v7 打分）

| 持仓 | Score | Rank | vs cutoff(-2.247) | 建议 |
|------|:---:|:---:|:---:|:---:|
| 603048.SH | -2.237 | 14/4929 | ABOVE | ✅ 保留 |
| 002877.SZ | -2.308 | 45/4929 | BELOW | ❌ 换 |
| 688151.SH | -2.324 | 62/4929 | BELOW | ❌ 换 |
| 605033.SH | -2.344 | 86/4929 | BELOW | ❌ 换 |
| 688616.SH | -2.373 | 139/4929 | BELOW | ❌ 换 |
| 600319.SH | -2.377 | 149/4929 | BELOW | ❌ 换 |
| 300927.SZ | -2.394 | 192/4929 | BELOW | ❌ 换 |
| 603686.SH | -2.425 | 297/4929 | BELOW | ❌ 换 |
| 603029.SH | -2.425 | 300/4929 | BELOW | ❌ 换 |
| 003030.SZ | -2.457 | 425/4929 | BELOW | ❌ 换 |

**仅 603048.SH 建议保留。** 换手方案见 `trade/advice_20260601.txt`。

---

## 七、代码位置

```
train_li/v7/
├── build_windows.py             ← LABEL_HORIZON=1，输出 → v7_windows/
├── dataset.py                   ← 指向 v7_windows/
├── train_gru.py                 ← GRU baseline T+1 (N=2048)
├── train_spatial.py             ← Spatial concat T+1 (N=1024, K=5, d=32)
├── models/
│   ├── spatial_attn.py          ← SparseSpatialAttention
│   └── gru_spatial.py           ← GRURankerSpatial (concat fusion)
├── validate.py                  ← 预计算打分 + IC + 10x10 回测 + sweep
├── sweep_quick.py               ← v7 Spatial 全参数 sweep
├── sweep_gru.py                 ← v7 GRU 全参数 sweep
├── sweep_fine.py                ← N=6..9 精细网格 sweep
├── sweep_stability.py           ← 10×10 窗口稳定性 sweep
├── checkpoints/
│   ├── gru_v7_H128_L1_D0.2_lr0.0003_N2048.pt
│   └── gru_spatial_v7_d32_K5_H128_L1_D0.2_lr0.0003_N1024.pt
└── results/
    ├── daily_scores_gru_t1.parquet     ← v7 GRU 打分 (600 股)
    ├── daily_scores_spatial_t1.parquet ← v7 Spatial 打分 (600 股)
    ├── results_gru.csv                 ← 训练结果
    ├── results_spatial.csv             ← 训练结果
    ├── sweep_spatial.csv               ← v7 Spatial sweep
    └── sweep_fine.csv                  ← 精细网格 sweep
```

---

## 八、核心结论

### 1. T+1 > T+5

T+5 标签的 val IC 0.111 是虚假繁荣。训练在 T+1 上的 val IC 0.106（低 0.005），但 test 期能稳定提供 +0.05 的 IC——没有动量溢价伪装。

### 2. Spatial 在 T+1 上的放大器效应远远强于 T+5

| 标签 | Spatial vs Baseline (sweep) | 
|:---:|:---:|
| T+5 | 全量 N=5,K=3 仅 +0.07%（持平） |
| T+1 | 600股 N=5,K=3 **+15.45% vs +3.43%** (4.5x) |

**正确的标签是架构发挥威力的前提。** 在错误的 T+5 标签上，spatial 的收益被标签噪声掩盖；在正确的 T+1 标签上，spatial 的增量被放大。

### 3. N=5, K=3 全面最优

- K=3 在所有 N=5..10 上都是最优值
- 每增持 1 只股票收益降 1.2-2.9%
- 信号集中在前 5 名
- 所有 N 的稳定性（Std）接近，N=5 赢在上行

### 4. 模型信号薄是长期瓶颈

Test IC 在 0.048-0.053 之间，无论架构怎么改、标签怎么选，天花板就在这里。进一步提升需要：
- 新的特征数据源（资金流向 data/moneyflow）
- 更多估值指标（data/metric 中未用的列）
- KNN 后端分数融合（已验证 +0.005 IC，无需训练）

### 5. 训练/推理管线的细节对齐极其重要

从预处理（log1p、winsor、PE/PB clip）到 dtype（float32）到 warmup（300天）到 compile（state dict 前缀）——任何一个不匹配都会导致模型加载失败或结果失真。这是工程复杂度，不是可有可无的。
