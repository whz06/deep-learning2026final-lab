# 最终实验报告方案

> 创建: 2026-06-15  
> 状态: 唯一事实依据，已确认  
> 组员: PB24050961-李沐泽 / PB24030836-吴鸿哲 / PB24261889-王俊涛

---

## 一、报告结构（7章 + 3附录）

```
第一章 项目背景与问题定义
    1.1 任务定义
    1.2 数据来源与股票池过滤
    1.3 预测标签构造
    1.4 数据集划分（严格时序，禁止随机打乱）
    1.5 交易成本模型
    1.6 评测指标体系（IC/ICIR/Sharpe/MaxDD/年化收益）

第二章 数据预处理与特征工程
    2.1 共同前置处理（ST/北交所/停牌/新股过滤）
    2.2 路线A特征工程 [Li]
        2.2.1 26维特征体系（量价+技术+截面+估值）
        2.2.2 两阶段归一化（per-window Z-score + per-date Z-score）
        2.2.3 预处理增强（log1p, winsorization, PE/PB clip）
    2.3 路线B特征工程 [Wu]
        2.3.1 因果Max-Min归一化设计
        2.3.2 Sigmoid全局归一化及其系统性失效分析
        2.3.3 滑动窗口构造（T=20）
    2.4 数据泄露防护 [★显式强化]
        路线A: 两阶段归一化（V5修复截面特征零化bug）→ 时序/截面分别归一化
        路线B: 因果Max-Min（只用截至当日的历史极值）→ 无未来信息
        共同: 训练/验证按时间切分不随机打乱, per-stock级别切分

第三章 路线A — 排序预测路线（Li，详细论述，V1-V9完整叙事）
    3.1 技术路线总览：Learning to Rank 范式
    3.2 基线建立（V1-V2）
    3.3 交易策略设计与风控（V3 — Strategy B确立）
    3.4 预处理体系重构（V5 — 最大突破 +8.2%）
    3.5 空间注意力机制（V6 — d=32 bottleneck 设计）
    3.6 标签修正（V7 — T+5→T+1，诚实信号）
    3.7 失败扩展尝试（V8-V9）
    3.8 路线A独立评测结果
        3.8.1 训练/验证损失曲线
        3.8.2 各版本验证IC演化
        3.8.3 统一回测（V6 vs V7 GRU vs V7 Spatial, 含成本）
        3.8.4 月度IC与收益分解
    3.9 版本迭代核心经验表

第四章 路线B — 价格预测路线（Wu，适度扩展）
    4.1 技术路线总览：价格回归→反算收益→排序选股
    4.2 模型架构：CNN-LSTM
        4.2.1 架构详解（3层Conv1d + 2层LSTM(256)）
        4.2.2 参数量分析（865K, CNN仅占1%）
    4.3 归一化方案对比研究 ★核心方法论贡献
        4.3.1 Max-Min因果累积：设计原理
        4.3.2 Sigmoid全局归一化：失效机制深度分析
        4.3.3 对比实验（IC=0.118 vs IC=-0.016）
    4.4 训练策略探索
        4.4.1 从单年到3年混合训练
        4.4.2 参数/样本比分析（3.67× → 0.45×）
        4.4.3 早停与训练过程IC稳定性
    4.5 路线B独立评测结果
        4.5.1 各配置IC/IR/RankIC对比
        4.5.2 12天周期IC分解
        4.5.3 N-K参数分析（K无影响的独特发现）
    4.6 统一回测结果（N=5 K=3, 含成本, Strategy B）

第五章 统一评测对比
    5.1 统一评测框架说明
        测试期: 2026-01-05→05-29 (94天)
        成本: 净值法 (卖0.076%+买0.026%)
        策略: Strategy B (CSI5d<-1%→80%仓)
    5.2 路线A内部模型对比 ★详细定量对比
        5.2.1 信号质量对比（V6 vs V7 GRU vs V7 Spatial）
        5.2.2 N×K 策略表现对比
        5.2.3 风险与稳定性对比
    5.3 路线A vs 路线B —— 收益率对比
        5.3.1 统一回测收益对比表
        5.3.2 累计收益曲线叠加（含CSI300基准）
    5.4 方法论维度对比
        5.4.1 预测范式：Learning to Rank vs Price Regression
        5.4.2 特征哲学：多维融合 vs 极简单变量
        5.4.3 架构哲学：特征驱动 vs 模型驱动
    5.5 两条路线的互补性与融合方向

第六章 模拟交易竞赛（2026.6.1-12）
    6.1 竞赛规则与操作流程
    6.2 竞赛表现（使用路线A模型，代表全队）
        6.2.1 收益趋势图 [APP截图]
        6.2.2 调仓记录 [APP截图]
        6.2.3 持仓记录 [APP截图]
    6.3 竞赛总结与反思

第七章 总结与展望
    7.1 核心经验（6条）
    7.2 模型局限性
    7.3 未来方向

附录A 组员分工（待填）
附录B 图表索引
附录C 代码结构与复现指南
```

---

## 二、图表清单

### 路线A图表（Li — 现有 17 张）

| 编号 | 内容 | 文件 |
|:---:|------|------|
| 图3.1 | V2三模型验证IC对比 | fig1_model_comparison.png |
| 图3.2 | 各版本验证IC演化 | fig2_version_ic_evolution.png |
| 图3.3 | Strategy A/B/C对比 | fig3_strategy_comparison.png |
| 图3.4 | 窗口对比 | fig4_window_comparison.png |
| 图3.5 | KNN Proxy K-α IC增益热力图 | fig5_knn_proxy_heatmap.png |
| 图3.6 | V6空间注意力V1→V2修正效果 | fig6_spatial_v1_v2.png |
| 图3.7 | 空间注意力K值扫描 | fig7_k_sweep.png |
| 图3.8 | Cap IC分析 | fig8_cap_ic.png |
| 图3.9 | V7 Spatial N×K扫参热力图 | fig9_nk_sweep_heatmap.png |
| 图3.10 | 月度收益分解 | fig10_monthly_decomposition.png |
| 图3.11 | 累计收益曲线（含回撤阴影） | fig11_cumulative_returns.png |
| 图3.13 | 损失函数对比 | fig13_loss_comparison.png |
| 图3.14 | 单因子IC贡献排名 | fig14_feature_ic_ranking.png |
| 图3.15 | V9 λ 扫描 | fig15_lambda_sweep.png |
| 图3.16 | Jan效应Alpha/Beta分解 | fig16_jan_decomposition.png |
| 图3.17 | 月度IC趋势 | fig17_monthly_ic_trend.png |

### 路线B图表（Wu — 需新增 3 张 + 复用 3 张）

| 编号 | 内容 | 来源 | 状态 |
|:---:|------|------|:---:|
| 图4.1 | CNN-LSTM架构示意图 | **需新增** | ❌ |
| 图4.2 | Max-Min vs Sigmoid IC对比 | **需从数据生成** | ❌ |
| 图4.3 | 训练损失对比（1年 vs 3年） | `loss_compare.png` | ✅ |
| 图4.4 | 3年模型训练IC演化 | 从 `maxmin_cnn_lstm_L20.json` 提取 | ❌ |
| 图4.5 | N参数收益影响 | **需新增** | ❌ |
| 图4.6 | Sigmoid vs 股价分布 | `sigmoid_vs_distribution.png` | ✅ |

### 对比图表（Ch5）

| 编号 | 内容 | 来源 | 状态 |
|:---:|------|------|:---:|
| 图5.1 | 路线A内部日IC时序叠加 | `comparison/fig1_daily_ic_timeseries.png` | ✅ |
| 图5.2 | 路线A内部滚动IC对比 | `comparison/fig2_rolling_ic.png` | ✅ |
| 图5.3 | 路线A内部IC直方图叠加 | `comparison/fig3_ic_histogram.png` | ✅ |
| 图5.4 | 路线A内部累计收益叠加 | `comparison/fig4_equity_curves.png` | ✅ |
| 图5.5 | 路线A vs B累计收益叠加 | **需生成** | ❌ |
| 表5.1 | 路线A vs B 统一回测收益对比 | 已有数据 | ✅ |
| 表5.2 | 方法论维度对比 | — | ✅ |

### 竞赛图表（Ch6）

| 编号 | 内容 | 来源 | 状态 |
|:---:|------|------|:---:|
| 图6.1 | 竞赛收益趋势 | 同花顺APP截图 | ❌ |
| 图6.2 | 调仓记录 | 同花顺APP截图 | ❌ |
| 图6.3 | 持仓记录 | 同花顺APP截图 | ❌ |

---

## 三、数据来源汇总

| 数据 | 路径 | 用途 |
|------|------|------|
| Li V6 打分 | `train_li/v6/results/daily_scores.parquet` | Ch3 + Ch5 |
| Li V7 GRU 打分 | `train_li/v7/results/daily_scores_gru_t1.parquet` | Ch3 + Ch5 |
| Li V7 Spatial 打分 | `train_li/v7/results/daily_scores_spatial_t1.parquet` | Ch3 + Ch5 (参考) |
| Wu N5K3 回测日收益 | `train_wu/CNN-LSTM/result/backtest/backtest_n5k3_daily.csv` | Ch4 + Ch5 |
| Wu N5K3 回测汇总 | `train_wu/CNN-LSTM/result/backtest/backtest_n5k3_summary.json` | Ch4 + Ch5 |
| Wu 训练metrics | `train_wu/CNN-LSTM/result/maxmin_cnn_lstm_L20.json` | Ch4 |
| Wu IC对比 | `train_li/report/comparison/daily_ic_all.csv` | Ch5 (标注为单年训练参考) |
| Wu 回测报告 | `train_wu/CNN-LSTM/result/backtest/backtest_n5k3_report.md` | Ch4 叙事参考 |
| Wu 实验报告 | `train_wu/CNN-LSTM/result/experiment_report.md` | Ch4 叙事参考 |
| Wu 数据分析 | `train_wu/CNN-LSTM/data_analysis.md` | Ch2 |
| 竞赛数据 | 同花顺APP → 截图 | Ch6 |

---

## 四、待完成工作清单

| # | 工作项 | 产出 | 优先级 |
|:---:|------|------|:---:|
| 1 | 从同花顺APP获取竞赛收益/调仓/持仓截图 | 图6.1-6.3 | 🔴 P0 |
| 2 | 生成CNN-LSTM架构示意图 | 图4.1 | 🟡 P1 |
| 3 | 从Wu JSON提取test_ic序列生成训练IC演化图 | 图4.4 | 🟡 P1 |
| 4 | 生成Max-Min vs Sigmoid IC对比图 | 图4.2 | 🟡 P1 |
| 5 | 生成N参数收益影响图 | 图4.5 | 🟡 P1 |
| 6 | 生成路线A vs B累计收益叠加图 | 图5.5 | 🟡 P1 |
| 7 | 撰写Ch4路线B扩展叙事（基于现有600行experiment_report重构） | text | 🟡 P1 |
| 8 | 撰写Ch5统一对比章节 | text | 🟡 P1 |
| 9 | 撰写Ch6竞赛章节 | text + 截图 | 🟡 P1 |
| 10 | 撰写Ch7总结 + 附录 | text | 🟢 P2 |
| 11 | 填充组员分工信息 | 附录A | 🟢 P2 |
| 12 | 最终LaTeX排版审查 | — | 🟢 P2 |

---

## 五、关键设计决策

1. **诚实性优先**：Wu模型在统一框架下收益为负（-4.01%），报告中如实呈现，分析"价格回归IC不能直接转化为排序收益"
2. **对比层级分明**：路线A内部（V6 vs V7 GRU vs V7 Spatial）详细定量对比；路线A vs B 只比最终收益率和方法论
3. **叙事完整性**：路线A保持 V1→V9 迭代叙事，路线B保持"归一化探索+训练规模教训"叙事，各有起承转合
4. **格式**：LaTeX (基于现有 `main.tex` 模板)
5. **数据泄露防护** 在 Ch2.4 显式强化
