# Step 3 & Step 4 实验报告 — 架构改进尝试

> 日期：2026-06-01 | 基线模型：v5 GRU (H=128, L=1, D=0.2, lr=3e-4, N=2048, 26-dim)

---

## 背景

v5 GRU baseline 在 val 上达到 **Rank IC = 0.1114**（T+5 label + log1p/winsor/PE-PB clip 预处理）。在此基础上尝试两种架构改进：

- **Step 4**: Temporal Attention Pooling — 用可学习权重对 GRU 60 个时间步做 softmax 加权求和，替换 `last hidden`
- **Step 3**: Sparse Spatial Attention — 每只股票通过 learnable Q/K 关注 K 个最相似邻居，用邻居信息增强自身表征

---

## 训练配置

| 参数 | 值 |
|------|-----|
| 数据划分 | Train: 2019–2024, Val: 2025 |
| 特征维度 | 26 (19 temporal + 7 cross/valuation) |
| 标签 | T+5 累计收益率 |
| GRU 配置 | H=128, L=1, D=0.2 |
| 优化器 | AdamW, lr=3e-4, weight_decay=1e-5 |
| 损失函数 | ListMLE |
| 验证指标 | Spearman Rank IC |
| val_every | 2 |
| PATIENCE | 7 |
| N_sample | 2048 |
| AMP | True |
| torch.compile | Step 4: True, Step 3: False (Triton 不可用) |

---

## Step 4: Temporal Attention Pooling

### 设计

在 GRU 输出 `out [N, T, 128]` 上做时序注意力：

```python
w = softmax(Linear(128, 1)(out), dim=1)  # [N, T, 1]
pooled = (out * w).sum(dim=1)            # [N, 128]
```

仅增加 129 个参数。如果 softmax 退化到仅关注 t=60，则等效于 baseline GRU（无退化风险）。

### 结果

| 指标 | 值 |
|------|-----|
| Best Val Rank IC | **0.1072** |
| Best Epoch | 2 |
| Total Epochs | 16 (early stop) |
| 训练时间 | 7.2 min |
| 参数量 | 68,354 |
| vs Baseline | **-0.0042 (-3.8%)** |

### 结论

❌ **未提升**。GRU 的门控机制已能自适应保留关键时序信息，叠加可学习 attention pooling 引入额外参数但无新增信息源，导致轻微过拟合。GRU last hidden 在该任务上已经是最优时序聚合方式。

---

## Step 3: Sparse Spatial Attention (K=10)

### 设计

GRU 输出 `h [N, 128]` 后，每只股票通过 Q/K projection 计算与所有股票的内积相似度，取 Top-K 最近邻（排除自身），用 softmax 归一化后的注意力权重聚合邻居的 value vectors，残差连接回原始表征：

```
h_enhanced = h + attention_pool(KNN(Q(h), K(h)), V(h))
```

参数量：49,024（3 个 Linear(128,128)）。复杂度 O(NK) 而非 O(N²)。

### 前置证据

v4/verify_spatial.py 用 KNN proxy（固定截面特征选邻居 + 简单平均）在 test 期获得 **+0.0049 IC 增益**（0.0424 → 0.0474，+11.5%）。如果 learnable attention 能学到类似的邻居关系，预期 val IC 应达到 ~0.116+。

### 结果

| 指标 | 值 |
|------|-----|
| Best Val Rank IC | **0.1086** |
| Best Epoch | 4 |
| Total Epochs | 18 (early stop) |
| 训练时间 | 8.3 min |
| 参数量 | 117,377 |
| vs Baseline | **-0.0028 (-2.5%)** |

### 结论

❌ **未提升**。Learnable Q/K 投影后学到的相似度空间与 KNN proxy 中固定的截面特征空间（pct_chg/amount/turnover_rate）完全不同。KNN proxy 的增益来自对 GRU 分数的简单加权平均（denoising effect），而非真正发现了互补的 alpha 信息来源。Learnable attention 在训练过程中无法自然收敛到这种简单平均行为。

---

## 汇总

| 模型 | Val IC | Δ | 参数 | 时间 | 判断 |
|------|:------:|:--:|:----:|:----:|:----:|
| v5 GRU baseline | **0.1114** | — | 68,225 | 18 min | ✅ 当前最优 |
| + Temporal AttnPool (Step 4) | 0.1072 | -3.8% | 68,354 | 7 min | ❌ |
| + SparseSpatialAttn (Step 3) | 0.1086 | -2.5% | 117,377 | 8 min | ❌ |

---

## 关键数据

### Step 4 训练日志摘要

```
ep   2V | val_ic 0.1072 | best 0.1072 @2    ← 最优
ep   4V | val_ic 0.1062
ep   6V | val_ic 0.1038
ep  10V | val_ic 0.1028
ep  16  | Early stop (7 epochs no improvement after ep 9)
```

### Step 3 训练日志摘要

```
ep   2V | val_ic 0.1057 | best 0.1057 @2
ep   4V | val_ic 0.1086 | best 0.1086 @4    ← 最优
ep   8V | val_ic 0.1067
ep  10V | val_ic 0.1045
ep  18  | Early stop (7 epochs no improvement after ep 11)
```

### 基线 v5 GRU（参考）

```
v5.1 (T+1, 无预处理):  best IC = 0.1030 @ epoch 27
v5.2 (T+5, 预处理):    best IC = 0.1114 @ epoch 9   ← 当前 baseline
v2 GRU (T+1, 22-dim):  best IC = 0.1029
```

---

## 代码位置

```
train_li/v6/
├── models/
│   ├── gru_attn.py          ← Step 4: GRURankerAttn
│   ├── spatial_attn.py      ← SparseSpatialAttention module
│   └── gru_spatial.py       ← Step 3: GRURankerSpatial
├── train_step4.py           ← Step 4 训练脚本
├── train_step3.py           ← Step 3 训练脚本
├── checkpoints/
│   ├── gru_attnpool_*.pt    ← Step 4 checkpoint
│   └── gru_spatial_*.pt     ← Step 3 checkpoint
└── results/
    └── results.csv          ← 训练结果

train_li/v5/train.py         ← val_every→2, PATIENCE→7
train_li/trade/infer.py      ← 添加 log1p + winsorization + PE/PB clip
```

---

## 后续方向

1. **KNN proxy 的简单平均也许可以直接用**：在推理时对 GRU 分数做 KNN 平均（无需训练），veto/confirm 是否能复现 +0.0049
2. **调整 K 值**：当前仅试了 K=10。K=3/5 在 proxy 测试中也正向，可能更稳定
3. **其他 loss**：ListMLE 可能已充分优化排序，架构改动在强 baseline 上难以体现增量
4. **策略层面**：Step 5（Adaptive Strategy B）不依赖训练，可直接做 backtest 验证
