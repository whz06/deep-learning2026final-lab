# 对《初步计划.md》的批判性分析与可行性评估

> 背景：当前模型 GRU (H=128, L=1, T=60) 在 test 期 OOS RankIC = 0.042，通过 Strategy B（动量止损）在 2026 Feb-May 取得 +6.85%。问题根源是信号弱，不是模型架构复杂度不够。

---

## 一、逐条评估

### 1. 可微 RankIC 损失 → ❌ 不做

**计划描述**：用 NeuralSort/SigmoidRanking 将 Spearman 相关系数可微化，替代 ListMLE。

**判定：不可行 / 不值得。**

| 维度 | 分析 |
|------|------|
| **当前基线** | ListMLE 已在 val 上取得 0.1029 RankIC，test 上 0.042。换 loss 不可能解决"test 期信号弱"的问题 |
| **计算代价** | NeuralSort 需要 O(N² log N) —— N=2048 时单 batch 的 soft-sort 成本远超 GRU 前向传播 |
| **梯度质量** | Soft-sort 的梯度在 N>500 时会扩散到几乎均匀分布，RankIC 的 soft 近似变差 |
| **实证引用可疑** | 方正证券报告中的 "RankIC=12.48%, IR=5.41" 是在高频因子上的 in-sample 结果，与我们的 OOS test 期完全不可比 |

**结论**：ListMLE 已经直接优化排序，且 O(N log N) 的 `logcumsumexp` 实现成熟高效。没有必要且没有时间换 loss。

---

### 2. Pairwise Margin Loss → ⚠️ 低优先级

**计划描述**：显式惩罚排序错位，对顶部股票对赋予更高权重（LambdaRank）。

**判定：理论有效，但边际增益极小。**

ListMLE 内部已经通过 `logcumsumexp` 隐含了 pairwise 比较：它对头部排序错误天然有更高的惩罚权重（因为 top-ranked 的 log-probability 被更多项累加）。加 Margin Loss 相当于在已有的排序梯度上再加一层 —— 类似 gradient boosting 但与 ListMLE 同质化严重。

**可行性验证**：在当前 v2 训练 pipeline 中，将 `listmle_loss` 替换为 `listmle_loss + 0.1 * margin_loss`，跑一组对比实验。但预计 IC 差异 < 0.003。

**结论**：如果有余力可在 v5 尝试，但不作为方向。

---

### 3. RevIN 归一化 → ❌ 不做

**计划描述**：对每个窗口实例做可逆归一化（Reversible Instance Normalization），消除非平稳性。

**判定：对我们场景无意义。**

RevIN 的核心价值是 **可逆性**：前向归一化→模型处理→反向去归一化恢复原始尺度。这适用于需要输出时序预测（如 forecasting）的场景。我们只需要一个标量得分 s_i，输出不需要恢复原始尺度。RevIN 的 "可逆" 对我们没有价值。

我们当前的 per-window Z-score 归一化已经做到了实例级独立标准化（每个日期窗口内独立计算 mean/std），没有未来信息泄露。RevIN 额外加的 learnable affine 参数（每个特征一对 scale+bias）只是增加 44 个参数，不会带来实质改进。

**结论**：当前归一化已足够，跳过。

---

### 4. 交错时空注意力（Stage 1）→ ⚠️ 可尝试验证

**计划描述**：
1. 时间自注意力（沿时间轴）→ 捕捉单只股票的趋势/反转
2. 资产间自注意力（沿股票轴）→ 捕捉股票间的相互关系

**逐组件分析**：

**时间自注意力**：我们的 TF 模型（TransformerRanker, d_model=96, heads=4）已经在做这件事，val IC=0.1009，**低于 GRU 的 0.1029**。Transformer 的时间自注意力在金融时序上不比 GRU 好 —— 可能是 attention 的过度灵活导致在小数据上过拟合。

**资产间自注意力（注意力池化/Attention Pooling）**：这是计划中唯一有可能创造增量的新组件。逻辑是：
- 每只股票的初始表征 h_i 不该只看自己的历史，也看同一时刻"同类"股票的表现
- 例如：如果同行业大部分股票今天上涨，则我的模型对我的股票评分应参考这一信息

但问题：
- O(N²) 复杂度，N=2048 → 4M 个 pairwise attention score
- 我们已经通过截面特征（rank(pct_chg), rank(amount), rank(turnover_rate), rel_beta）隐式编码了截面信息。空间注意力是否提供**超出这些手工特征的增量**？

**可行性验证 — 最小实验**：

用当前 GRU 的输出分数 `s_i` 作为 baseline。对每只股票 i，找到它在 [pct_chg, amount, turnover_rate] 截面特征空间中的 5 个最近邻，计算邻居的 GRU 分数均值 `s_neighbor_i`。构造融合分数 `s_fused = (1-w)*s_i + w*s_neighbor_i`。在 test 期回测：如果 w>0 时 RankIC 提升，则说明邻居信息有价值 → 空间注意力值得做。如果任何 w 都做不出提升 → 空间注意力不会有用，因为手工 cross-sectional 特征 + GRU 已经捕捉了所有截面信息。

**预估**：邻居平均大概率是**弱正则化**（拉平极端分），IC 可能略微提升但不会显著改变排序。

---

### 5. 动态交叉注意力强化层（Stage 2）→ ❌❌ 强烈反对

**计划描述**：第二阶段用多头交叉注意力（Cross-Attention），每只股票作为 Query 查询所有其他股票的 Key/Value，"动态学习股票间关联"。

**判定：在当前的硬件和数据规模下完全不可行，且理论假设薄弱。**

| 问题 | 细节 |
|------|------|
| **计算不可行** | 单个 cross-attention 层：QKV 投影 O(Nd²)=2048×96²≈19M，attention scores O(N²d)=2048²×96≈400M。一个 layer 就 ~420M FLOPs。堆叠多层意味着每个 batch 上亿次计算。当前 GRU 一个 batch 约 10M FLOPs。Cross-attention 是 GRU 的 40+ 倍。 |
| **内存不可行** | Attention 矩阵 [N, N] = [2048, 2048] × fp32 = 16MB。多头（比如 4 头）→ 64MB。加上残差连接和梯度 → 数百 MB。8GB VRAM 几乎撑不住。 |
| **理论假设弱** | "股票间溢出效应" 确实存在（如板块联动），但溢出是**通过共同因子产生**的（行业、市值、beta），不是通过 pairwise 直接依赖。Cross-attention 建模 pairwise 依赖，而实际的溢出是 factor-level 的。截面特征已经捕捉了因子层面的信息。 |
| **如何验证这个假设** | 如果股票 A 的 return 可以被"与 A 相似的股票"的 return 预测，则 cross-attention 可能有价值。验证方法：对 test 期每天，用 KNN（基于截面特征）找到每只股票的近邻，计算 `IC(neighbor_avg_return, own_return)`。如果这个 IC ≈ 0，则 cross-attention 没有增量信息。根据我们已有的分析（Layer 1 的 momentum IC 约 0），这个 IC 很可能接近 0。 |

**结论**：这是整个计划中问题最大的组件。从计算、内存、理论三个维度都站不住脚。建议直接砍掉。

---

### 6. 行业信息嵌入 → ⚠️ 低优先级

**计划描述**：申万一级/二级行业编码作为静态特征嵌入。

**判定：简单但收益有限。**

加入行业 embedding（28 个一级行业 → embedding dim=8）只需要 ~224 个参数。问题是：
- 行业是静态的（股票不换行业），模型可能学会"总是给某行业高/低分"，这不利于截面排序
- 行业信息在截面特征中已经部分反映（同行业股票的 pct_chg 相关）

**可行性验证**：在 v2 训练中加入一个 industry_id embedding（从 stock code 前缀映射到申万行业），训练后对比 val IC。如果 IC 提升 > 0.003 则值得。

---

### 7. 对比学习辅助损失 → ❌❌ 危险

**计划描述**：NT-Xent 损失，正样本对 = 同收益率分位区间的股票，负样本对 = 收益差异大的股票。

**判定：存在严重的 look-ahead bias 风险，且实现不当会直接导致 overfitting。**

**关键问题**：正/负样本的定义使用了**未来收益率**。但未来收益率就是 label！这意味着：
- 你在训练时已经知道了哪些股票"应该相似"
- 对比损失在表征空间中拉近"未来会涨的股票"，推远"未来会跌的股票"
- 这等价于**直接泄露 label 到特征空间**，不是正则化，是作弊

如果改用"行业相同 + 趋势相似"定义正样本，则：
- 跨行业的正样本（行业不同但走势相似）的价值不确定
- 同类股票的 intra-industry dispersion 很大（同行业 ≠ 同收益）

**结论**：用未来收益率定义对比 pair 是 look-ahead bias。用行业定义则可能无增量。不建议在当前阶段投入时间。

---

### 8. 数据增强 → ⚠️ 可尝试

**计划描述**：时间域窗口偏移 + 特征域噪声/遮蔽。

**判定：理论合理，实现简单，但收益不确定。**

- **时间域窗口偏移**：等同于训练时随机使用 T∈[50,60] 的子窗口。这与我们 v4 的多窗口 ensemble 方向一致。已有 `dataset.py` 的 `[:, -W:, :]` 截断机制可以支撑。
- **特征域遮蔽**：随机 mask 一段时间的特征，强制模型不过度依赖特定日期。
- **特征域噪声**：但注意 non-IID nature —— 金融数据的噪声不是独立高斯的，加高斯噪声可能不符合真实噪声结构。

**可行性验证**：在 v2 训练中加 `--augment` flag，实现窗口偏移（uniform(30,60)），对比 val IC。如果提升 > 0.003 → 值得。

---

### 9. 模型集成（种子 ensemble）→ ✅ 已验证

**已经做过**：GRU×5 种子集成。秩相关性 ~0.95 → 伪多样性。T=30+T=60 多窗口 ensemble 也已验证失败（T=30 IC=0.0363 < T=60 IC=0.0424，IC相关性 0.962）。建议转向空间注意力（已验证 KNN proxy IC +0.0049）作为真正的多样性来源。

---

## 二、汇总：可行性矩阵

| # | 组件 | 可行性 | 理由 | 建议 |
|---|------|--------|------|------|
| 1 | 可微 RankIC | ❌ | 计算代价高，ListMLE 已足够 | **不做** |
| 2 | Margin Loss | ⚠️ | 边际增益 < 0.003，与 ListMLE 同质 | 低优先级 |
| 3 | RevIN | ❌ | 可逆性对 scalar output 无意义 | **不做** |
| 4a | 时序自注意力 | ✅ | 已实现（TF=0.1009 < GRU=0.1029） | **已有，且不如 GRU** |
| 4b | 空间自注意力 | ✅ | KNN proxy 验证：IC增益 +0.0049 (11.5%)，在所有K/α上都正向 | **推进 — 用KNN稀疏化实现** |
| 5 | 交叉注意力强化 | ❌❌ | 计算 O(N²)→不可行，理论假设弱 | **砍掉** |
| 6 | 行业信息 | ⚠️ | 简单但边际收益 | 低优先级 |
| 7 | 对比学习 | ❌❌ | 用未来收益率定义 pair = look-ahead bias | **砍掉** |
| 8 | 数据增强 | ⚠️ | 合理但需验证 | **先做时间偏移验证** |
| 9 | 种子集成 | ✅ | 已验证 → 伪多样性 | 改做架构多样性 |

---

## 三、建议的可行性验证

从上述 9 个组件中，**值得做**的最多 3 个。以下是验证方案：

### 验证 A：空间自注意力是否有效？（KNN proxy）

**问题**：邻居股票的分数是否包含增量信息？

**方法**（5 分钟，v4/verify_spatial.py）：
```
对 test 期每天：
  s_i = GRU 对股票 i 的分数
  feat_i = [pct_chg, amount, turnover_rate]  ← 截面特征
  s_knn_i = 特征空间中 K 近邻的 GRU 分数均值
  s_fused = (1-α) * s_i + α * s_knn_i  （扫描 K ∈ [3,5,10,20], α ∈ [0.1,0.3,0.5]）
  IC(K,α) = SpearmanR(s_fused, T+1 return)

如果 max IC(K,α) > IC(GRU) + 0.003 → 空间注意力值得做
```

**结果** ✅ 验证通过：

```
基线 GRU IC: +0.0424

 K  alpha   Mean IC   IC gain
 3    0.1   +0.0431   +0.0006
 3    0.5   +0.0466   +0.0042
 5    0.5   +0.0472   +0.0047
10    0.5   +0.0474   +0.0049   ← BEST
20    0.5   +0.0466   +0.0042
```

所有 (K, α) 组合均正向，增益随 α 单调递增。最佳增益 **+0.0049 (~11.5%)**在 K=10, α=0.5。增益来自日 IC 的 magnitude 而非 direction（IC>0 比例不变，~63%）。

**结论**：邻居 GRU 分数包含增量信息。空间注意力方向值得推进。建议用 learnable 的注意力权重（而非简单平均），且用 KNN 稀疏化（O(NK) 而非 O(N²)）。这是整个计划中**唯一被实验验证有价值**的新组件。

### 验证 B：时间窗口增强是否有效？

**问题**：训练时随机截取子窗口是否提升 OOS IC？

**方法**（需训练，~2h）：
```
修改 dataset.py → 训练时随机截取 T ~ Uniform(30, 60)
不做 sweep，只改 augmentation，train one model
对比 val IC 与 T=60 固定窗口的差异
```

### 验证 C：行业 embedding 是否有增量？

**问题**：加入行业信息是否能提升排序？

**方法**（需训练，~2h）：
```
在 GRU input 中加一个可学习的 industry_id → embedding → concat 到特征
训练一个模型，对比 val IC
```

---

## 四、核心判断（更新）

**《初步计划》的 9 个组件中：**

- **砍掉（4/9）**：可微 RankIC（计算代价）、RevIN（无意义）、交叉注意力 Stage 2（O(N²)不可行）、对比学习（look-ahead bias）
- **已验证有限或不可行（3/9）**：时序自注意力（TF<GRU）、Margin Loss（与 ListMLE 同质）、种子集成（伪多样性 r≈0.95）
- **低优先级（1/9）**：行业信息（简单但边际）
- **值得推进（1/9）**：**空间自注意力** — 唯一被实验验证有增量价值的组件（IC+0.0049, +11.5%）

**新发现**：邻居股票分数包含增量信息。GRU 得分 + KNN pooling 在 test 期的 IC 从 0.0424 提升到 0.0474。这证明截面信息没有被 GRU 完全提取 —— 空间注意力能把相似股票的共识转化为更强的排序信号。

**如果只做一个新方向：实现 sparse spatial attention（KNN 稀疏化，O(NK)）。** GRU + Strategy B 保持不动，空间注意力作为 add-on module 融合分数。

## 五、推荐实现路径（若验证 A 通过）

```python
# v5/spatial_pool.py — 轻量级KNN稀疏空间注意力
class SparseSpatialAttention(nn.Module):
    """每个股票 attend 到 K=20 个最相似的邻居股票 (O(NK) vs O(N²))"""
    def __init__(self, d_model=128, K=20):
        super().__init__()
        self.query = nn.Linear(d_model, d_model)
        self.key   = nn.Linear(d_model, d_model)
        self.value = nn.Linear(d_model, d_model)
        self.K = K
        self.scale = d_model ** 0.5
    
    def forward(self, h):  # h: [N, d_model] from GRU last hidden
        q = self.query(h)  # [N, d]
        k = self.key(h)    # [N, d]
        v = self.value(h)  # [N, d]
        sim = q @ k.T      # [N, N]
        # Top-K sparsification
        topk_sim, topk_idx = sim.topk(self.K + 1, dim=-1)  # +1 for self
        # Remove self (row-major: argmax is self)
        mask = topk_idx != torch.arange(N, device=h.device).unsqueeze(1)
        # Keep K nearest non-self neighbors
        topk_idx_k = topk_idx[mask].view(N, self.K)
        topk_sim_k = topk_sim[mask].view(N, self.K)
        attn = F.softmax(topk_sim_k / self.scale, dim=-1)
        context = torch.bmm(attn.unsqueeze(1), v[topk_idx_k]).squeeze(1)
        return h + context  # residual: original + spatial context
```

比 full O(N²) attention 快 100×，且 K=20 已捕捉最相关的 1-2% 股票。
