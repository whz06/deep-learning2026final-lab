# V6 实验全记录 — 空间注意力与架构尝试

> 时间：2026-06-01 | 基线：v5 GRU (T+5, H=128, L=1, D=0.2, 26-dim, val IC=0.1114)

---

## 一、背景与动机

v5 通过修复截面特征 bug + 预处理（log1p、winsorization、PE/PB clip）将 val IC 从 0.103 提到 0.111，但 test IC 仅 0.042-0.051。plan0601.md 规划了 Step 4（时序注意力池化）和 Step 3（稀疏空间注意力），尝试通过架构改进进一步提升信号强度。

前置证据：v4/verify_spatial.py 的 KNN proxy 验证显示，对 GRU 分数做基于截面特征的 K 近邻简单平均可在 test 期获得 **+0.0049 IC 增益**（0.0424 → 0.0474, +11.5%）。如果 learnable attention 能复现这个效果，有望进一步提升。

---

## 二、Step 4 — 时序注意力池化

### 设计

在 GRU 输出 `[N, T, 128]` 上做时序注意力：

```
w = softmax(Linear(128, 1)(out), dim=1)   // [N, T, 1]
pooled = (out * w).sum(dim=1)              // [N, 128]
```

仅增加 129 个参数。如果 softmax 退化到仅关注 t=60，则等效 baseline GRU（无退化风险）。

### 训练结果

| 指标 | 值 |
|------|:---:|
| Best Val IC | **0.1072** |
| vs Baseline (0.1114) | **-0.0042 (-3.8%)** |
| 训练时间 | 7 min |

### 结论

❌ **未提升。** GRU 的门控机制已能自适应保留关键时序信息，叠加可学习 attention pooling 引入额外参数但无新增信息源，导致轻微过拟合。

---

## 三、Step 3 — 空间注意力 (V1: 失败版)

### 设计

GRU 输出 `h [N, 128]` 后，通过 Q/K/V（均为 Linear(128, 128)）学习股票间相似度，Top-K 稀疏化，残差连接：

```
h_enhanced = h + attention_pool(KNN(Q(h), K(h)), V(h))
```

参数量 ~49K。K=10, N_sample=2048。

### 训练结果

| 指标 | 值 |
|------|:---:|
| Best Val IC | **0.1086** |
| vs Baseline (0.1114) | **-0.0028 (-2.5%)** |

### 结论

❌ **未提升。** Learnable attention 没有复现 KNN proxy 的增益。

---

## 四、诊断与根因分析

### 对比 SPATIAL_ATTENTION_PLAN.md 发现三个关键偏差

| | 原始计划 | V1 实现 | 后果 |
|---|---|---|---|
| Q/K 投影维度 | **d=32**（bottleneck） | d=128（=hidden_size） | 128 维全空间计算相似度，无关维度淹没信号 |
| 融合方式 | concat / gated / residual **sweep** | 仅 residual | 没测最优融合 |
| Self-exclude | `sim[i,i] = -inf` | topk(K+1) 手动剔除 | 边缘情况不稳定 |
| Projection | `h + proj(context)` 32→128 | `h + context` 无投影 | 噪声等权注入 hidden state |

**核心错误：d=128 bottleneck 缺失。** 这是标准 attention bottleneck 设计——d < d_model 强制模型选择对"找邻居"最关键的维度。d=128 时 Q/K 投影近乎恒等，similarity = h @ h^T，学习不到有意义的邻居关系。

### KNN proxy 为何能成功

KNN proxy 用**固定截面特征**（pct_chg/amount/turnover_rate）选择邻居，本质是 non-parametric denoising：对相似的股票取平均，抹掉个股噪声。learnable Q/K 学到的是完全不同的相似度空间，无法自然收敛到这种简单平均行为。

---

## 五、Step 3 V2 — 修正版（按原始计划）

### 修正内容

| 修正项 | V1 | V2 |
|--------|:---:|:---:|
| d_proj | 128 | **32** |
| Self-exclude | topk+remove | **sim[i,i] = -inf** |
| Fusion | 仅 residual | **concat / gated / residual sweep** |
| Projection | 无 | **Linear(32, 128)** |

### Phase 1 训练结果（d=32, K=10, N=1024）

| ID | 融合 | Val IC | vs Baseline |
|----|------|:------:|:-----------:|
| S1 | concat | **0.1126** | +1.1% |
| S2 | gated | **0.1122** | +0.7% |
| S3 | residual | **0.1131** | **+1.5%** |

✅ **三种融合全部超越 baseline！** d=32 bottleneck 是决定性因素。

**对比同融合方式（residual）的修正效果：**

| 版本 | d | proj | IC |
|------|:---:|:----:|:---:|
| V1 | 128 | ❌ | 0.1086 (-2.5%) |
| V2 | **32** | ✅ | **0.1131 (+1.5%)** |
| 增量 | — | — | **+0.0045** |

### Phase 2 — K 值扫描

取 S1(concat) 和 S3(residual) 扫 K=5, 10, 20：

| 融合 | K=5 | K=10 | K=20 |
|------|:---:|:---:|:---:|
| S1 concat | **0.1134** 🏆 | 0.1126 | 0.1111 |
| S3 residual | 0.1124 | 0.1131 | 0.1129 |

**冠军：S1 concat, K=5, d=32, N=1024: IC = 0.1134 (+1.8%)**

### Phase 3 — 全规模优化

| 配置 | Val IC | 结果 |
|------|:------:|:---:|
| N=2048, d=32, K=5 | 0.1099 | ❌ 退化 |
| N=2048, d=64, K=5 | 0.1104 | ❌ 退化 |

**N=2048 全规模退化。** 更多的股票 → 更多的 noise neighbor → 空间注意力更难学习有效关系。

### 最优模型

**S1 concat, K=5, d=32, N=1024：val IC=0.1134**

checkpoint: `v6/checkpoints/gru_spatial_concat_d32_K5_H128_L1_D0.2_lr0.0003_N1024.pt`

---

## 六、Test 期验证

### T+1 Test Rank IC（计算自 v6/precompute_backtest.py，全量 4929 只股票）

| 模型 | Test T+1 IC | IC>0 |
|------|:---:|:---:|
| v6 Spatial | **+0.051** | 67.1% |
| v2 GRU (参考) | +0.042 | 64.0% |

Spatial 在 test 期 IC 提升 21%。

### 按月 IC 分解（2026 Feb-May）

| 月 | Mean IC | IC>0 | 评价 |
|------|:---:|:---:|------|
| 2 月 | +0.10~+0.15 | 75-80% | 🟢 强 |
| 3 月 | +0.08~+0.17 | 67-100% | 🟢 强 |
| 4 月 | -0.29~+0.07 | 0-87% | 🟡 衰退 |
| 5 月 | -0.07~+0.05 | 33-71% | 🔴 失效 |

最优 5 日窗口（3/11-17）IC=+0.178，最差窗口（4/30-5/11）IC=-0.121。

### T+5 Test IC（模型训练的标签）

**T+5 IC = 0.010, IC>0=42.6%。** 模型训练在 T+5 上，但 test 期的 T+5 预测几乎无效——低于随机。这是一个重大发现，直接引出了 V7 的动机。

### IC by Market Cap

| 市值分位 | IC | IC>0 |
|:---:|:---:|:---:|
| 大 (top 1/3) | +0.044 | 57.5% |
| 中 | +0.057 | 63.0% |
| 小 (bot 1/3) | **+0.060** | **68.5%** |

小盘信号反而更强更稳定，且 v6 模型选股 97.7% 是大盘股——模型偏好和信号分布不匹配。

### Strategy Sweep（全量 4929 只，Strategy B）

最优：N=5, K=3 → 累计 +0.07%（几乎持平），但所有组合均跑输 CSI300（+6.43%）。

---

## 七、关键问题与教训

### 问题 1：d=128 导致 V1 失败

第一次实现时忽略了原始计划中 d=32 的关键设计参数，直接用 full hidden dim。这是标准 attention bottleneck 的核心设计原则——压缩投影维度强制学习有意义的相似度度量。

### 问题 2：torch.compile 在 Windows 上的 Triton 依赖

空间注意力模型的 forward 包含 topk/gather/bmm 操作，触发 PyTorch Inductor 的 Triton 后端，但 Windows conda 环境未安装 Triton。训练时必须 `--no-compile`。

### 问题 3：N_sample=2048 退化

更大的 sample size 理论上提供更丰富的截面比较，但实践中增加了噪声邻居，降低了空间注意力的信噪比。N=1024 反而是最优。

### 问题 4：T+5 标签的虚假繁荣

Val IC=0.1134 的训练结果在 test 期 T+5 上降到 0.010。模型实际学会的是 T+1 动量模式，T+5 上的高分只是因为训练期（2019-2025）动量延续到了 5 日范围。这是 V7 的出发点。

### 问题 5：trade/infer.py 预处理对齐

infer.py 的 v5 管线需要 log1p + winsorization + PE/PB clip 才能匹配训练特征。第一次运行全部 NaN 因为数据中存在缺失 OHLCV 值，需加 ffill() 处理。

### 问题 6：dtype float64 污染

CSI300 pct_chg 从 pandas DataFrame 读取后是 float64，与 cross_feat 的 float32 运算时自动提升为 float64，导致模型推理报 dtype mismatch。解决方案：全部 cast 到 np.float32。

---

## 八、代码位置

```
train_li/v6/
├── models/
│   ├── spatial_attn.py          ← SparseSpatialAttention (d=32 bottleneck)
│   ├── gru_attn.py              ← Step 4: Temporal AttnPool GRU
│   ├── gru_spatial.py           ← Step 3 V1: Spatial (d=128, 已废弃)
│   └── gru_spatial_v2.py        ← Step 3 V2: S1(concat)/S2(gated)/S3(residual)
├── train_step4.py               ← Step 4 训练
├── train_step3.py               ← Step 3 V1 训练
├── train_step3_v2.py            ← Step 3 V2 Phase 1-3 sweep
├── precompute_backtest.py       ← 全量打分 + 10x10 回测
├── diagnose_gap_v2.py           ← val→test IC gap 诊断
├── sweep_strategy.py            ← N×K 策略参数 sweep
├── analyze_cap.py               ← 市值分布分析
├── infer_v2.py                  ← v2 模型推理（22-dim 旧管线）
├── checkpoints/
├── results/
│   ├── phase1_spatial.csv       ← Phase 1-3 全部训练结果
│   ├── daily_scores.parquet     ← 74 天全量打分（363,693 条）
│   ├── strategy_sweep.csv       ← 策略参数 sweep
│   └── sweep_largecap.csv       ← 大中盘过滤 sweep
└── EXP_STEP3_STEP4.md           ← Step 3/4 初步实验报告
```

---

## 九、核心结论

1. **空间注意力有效但边际**：修正 d=32 后 val IC 从 0.1114 → 0.1134（+1.8%），test IC 从 0.042 → 0.051（+21%）。但这是"从弱到稍弱"，没有质变。

2. **架构改动天花板低**：所有模型改进最多贡献 +0.002 IC（~2%），特征工程（26-dim + 预处理）贡献了 +0.008（~8%）——特征 > 架构。

3. **T+5 是虚假信号**：训练在 T+5 上的模型在 test 期 T+5 上 IC=0.01（比随机差），但 T+1 上 IC=0.051。模型学会的是 T+1 动量，标签和能力的错配是 V5/V6 的核心问题。

4. **N_sample 不宜过大**：空间注意力的 N=2048 退化表明"更多股票 = 更多噪声"。N=1024 是最优点。

5. **修复缺陷比加新东西重要**：V1→V2 从一个 -2.5% 的失败变成 +1.5% 的成功，不是靠新架构，是靠修 bug（d=32、proj、-inf self-exclude）。
