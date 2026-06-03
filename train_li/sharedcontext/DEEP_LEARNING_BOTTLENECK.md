# Deep Learning A股排序模型困境 — 特征利用率不足

**背景**：A股 ~5000 只，每日排名预测下一日收益，top-N 策略 T+1 轮动。

---

## 一、当前状态

### 模型
- GRU(128, L=1) + SparseSpatialAttention(d=32, K=5 concat)，60日窗口
- ListMLE loss (listwise ranking)，批量 1024 股/日
- 输入 26 维：[19 temporal (OHLCV + tech) + 7 cross-sectional]

### 性能
| 指标 | 值 |
|------|-----|
| Val IC (2025) | 0.106 |
| Test IC (2026 Jan-May) | 0.048 |
| 回测净收益 (N=5, K=3) | +65.83% (94天) |
| 单因子 top IC | momentum_20d=0.060, amount=0.050, mf_flow_hhi=0.046 |

### IC 天花板
- 最佳模型 IC ~0.048-0.053
- 架构改动 (MLP→GRU→Spatial) gain ≤ +0.002
- 特征工程 (22→26→31·dim) gain ≤ +0.008
- **单个最好特征 IC=0.06，但模型 IC 卡在 0.048 → 多特征联合利用率不足**

---

## 二、已尝试且失败的方向

### 特征扩展 (V8)
- 新增 5 个特征：mf_flow_hhi, amihud_20, ret_skew_20, price_pos_20, mf_sm_lg_div
- 新增 moneyflow 19 维数据源
- 新增 111 行业 8-dim embedding + λ=0.1 空间注意力门控
- **结果**：val IC 0.1037 (vs v7 0.1062)，test IC 0.044 (vs v7 0.048)，回测 +1.80% (vs v7 +65.83%)
- **失败原因**：moneyflow 与 GRU 时序特征不兼容，行业 embedding 注入太浅

### 损失函数
- Weighted ListMLE: val IC = -0.067 (失败)
- LambdaRank: val IC = 0.007 (失败)
- **ListMLE 是唯一可用损失**，其他设计用于分类 relevance

### 减仓指标优化 (E1-E5)
- 5 步搜索最优 crash indicator → best = "永不减仓"（长期负期望）
- csi5d < -1% 止损：减仓后 55% 天数上涨 (net -5.30%)，2026 碰巧触发在真跌日
- 15 个市场特征与"5日跌 > 3%"最大相关性仅 +0.079

### 售卖模型 (V9 Phase 0)
- regime-conditional vol (牛市卖低波动 / 熊市卖高波动) 对所有 λ 均 ≤ v7 baseline
- 任何 λ > 0 叠加 vol 信号都造成回测净收益下降

---

## 三、核心困境

```
单因子 IC:   0.060  (momentum_20d)
             0.050  (amount)
             0.046  (mf_flow_hhi)
                   ↓
模型 IC:      0.048  (GRU+Spatial, 26-dim)
                   ↓
             瓶颈：模型无法将 3 个 IC>0.045 的单因子组合出 IC>0.053
                   ↓
解释：GRU 学到的模式 ≠ 单因子线性组合 → 非线性交互被 ListMLE 的全局排序梯度淹没
```

**根因假说**：
1. ListMLE 优化的是 1024/5000 全宇宙 rank likelihood，梯度分散在各 rank 位置，对小规模相对关系的区分度不够精细
2. GRU 60 天时序 → 单标量输出的信息压缩率太高 (60×26=1560 → 1)，中间 bottleneck 丢弃了大量特征交互信息
3. Spatial Attention K=5 concat 只做了 peer comparison 聚合，没有做 feature-level interaction
4. 30+ 个特征的交互被一个 128 维 hidden state 全权承担，表达能力不足

---

## 四、问 AI 的问题

1. 有哪些**不增加新特征、只改变特征在模型中的使用方式**就能提升 ranking IC 的方法？
   - 例如：feature gating, FiLM layer, feature-wise attention, cross-feature interaction, 混合专家 (MoE)
   
2. 在 ListMLE / listwise ranking 框架下，如何让模型**学到 fine-grained pairwise interaction** 而非被全局排序梯度淹没？
   - 例如：curriculum learning, hard negative mining, pairwise auxiliary loss, margin-based ranking

3. 时间序列 ranking 中，除了 LSTM/GRU + Attention 之外，还有哪些**信息压缩率更低**的架构？
   - 例如：TCN, Informer, 多尺度时序卷积 + 交叉特征网络

4. 对于金融股票排序，有没有论文或实践验证过 **feature interaction 层**（如 DCN v2 cross network, AutoInt, xDeepFM 等推荐系统方法）能在传统 LSTM/GRU 基础上稳定提升 IC？
   - 已知：GRU 提取时序表示，但 feature 维度之间的交叉被 MLP head 承担，表达能力有限

5. 多任务学习能否提升 ranking 质量？
   - 例如：同时预测 next-day return (ranking) + future volatility (regression) + sector relative ranking，共享 backbone

6. 有没有在金融排名场景下验证过 **contrastive learning** 或 **pairwise preference learning** 的效果优于 ListMLE？
   - 例如：直接构造 "stock A 明天比 stock B 好" 的 pairwise label，用 pairwise hinge / cross-entropy 训练

---

## 五、工程约束

- GPU: RTX 4060 8GB VRAM
- 训练: ~1456 天 (2019-2024)，每日 ~5000 股，batch=1 日，subsample 1024 股
- 推理: 每日 ~5000 股全宇宙打分，T+1 轮动
- 必须时间序列 split（不可随机打乱日期）
- 5000 股/天的 listwise 排序不是小规模检索排序，是 full-universe 金融 ranking
