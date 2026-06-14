# 实验报告完整计划

> 最后更新: 2026-06-14  
> 状态: Phase 1 收尾, Phase 2-4 待执行

---

## 一、报告目标

撰写一份符合课程要求（`作业要求.md`）的实验报告初稿，覆盖从 V1 到 V9 的完整探索历程。核心要求：

1. **详实的数据支撑**：所有 IC、回测数据来自实际结果文件，不凭空编造
2. **统一的评测框架**：所有模型在同一尺度下对比（同股票池、同日期、同成本模型、同策略）
3. **可视化表达**：关键数据用图表展示，便于理解
4. **失败分析**：记录哪些架构/方向被尝试并放弃，以及原因

---

## 二、统一评测框架（Phase 2-3）

### 评测矩阵

| 模型 | 打分文件 | 用途 |
|------|------|------|
| V6 Spatial (T+5) | `v6/results/daily_scores.parquet` | ✅ 已就绪 |
| V7 GRU (T+1) | `v7/results/daily_scores_gru_t1.parquet` | ✅ 已就绪 |
| V7 Spatial (T+1) | `v7/results/daily_scores_spatial_t1.parquet` | ⏳ Phase 1 收尾中 |
| V8 (T+1) | `v8/results/daily_scores_v8.parquet` | ⏳ Phase 1 收尾中（596股参考） |

统一参数：
- **测试区间**：2026-01-05 → 2026-05-29（94个交易日）
- **股票池**：各模型全量（V7 GRU/V8 标注"596股采样"）
- **策略**：Strategy B（csi5d < -1.0% → 80%）
- **成本模型**：净值法（净卖×0.076% + 净买×0.026%，分摊至N只）
- **V7 June**：2026-06-01 → 2026-06-12 作为独立竞赛附录

### 评测指标（Tier 1-4）

**Tier 1 — 信号质量（不依赖策略参数）**
- Mean Rank IC（日均 Spearman）
- IC Std（日间波动）
- ICIR（IC均值 / IC波动）
- IC > 0 比例
- 月均 IC（按 Jan-May 分解）

**Tier 2 — 策略鲁棒性（N×K 扫参）**
- 各模型最优 (N,K) 净收益
- N=5,K=3 锚点（公共参照）
- N=10,K=5 锚点
- 前5名平均
- 全网格均值

**Tier 3 — 风险与稳定性**
- Sharpe Ratio（vs CSI300 超额）
- Max Drawdown
- 月收益分解
- 10日滚动窗口胜率/最差窗口

**Tier 4 — 成本效率**
- Gross → Net 损耗
- 日均换手率
- 成本/超额比

---

## 三、数据文件状态

### 已完成（已 git commit）

| 文件 | 行数 | 股票 | 日期 | 状态 |
|------|:---:|:---:|:---:|:---:|
| `v6/results/daily_scores.parquet` | 466,700 | 4,940 | 95天 (Jan-May) | ✅ |
| `v7/results/daily_scores_gru_t1.parquet` | 467,604 | 4,947 | 95天 (Jan-May) | ✅ |

### Phase 1 待完成（在 Windows PowerShell 执行）

```powershell
cd D:\Workspace\DL_HW\deep-learning2026final-lab\train_li\report

# 1. V7 Spatial 全量 Jan-May
& D:\Software\miniconda3\envs\dl_lab1\python.exe score_models.py --model v7spatial --start 20260105 --end 20260529 --device cuda

# 验证
python -c "import pandas as pd; df=pd.read_parquet(r'..\v7\results\daily_scores_spatial_t1.parquet'); print(f'Rows:{len(df):,} NaN:{df[\"score\"].isna().sum()}')"

# 2. V7 Spatial June 1-12
& D:\Software\miniconda3\envs\dl_lab1\python.exe score_models.py --model v7spatial --start 20260601 --end 20260612 --device cuda

# 3. V8 全量 Jan-May（如果之前步骤验证通过）
& D:\Software\miniconda3\envs\dl_lab1\python.exe score_models.py --model v8 --start 20260105 --end 20260529 --device cuda
```

每个完成后在 WSL 中 git commit：
```bash
git add train_li/v7/results/daily_scores_spatial_t1.parquet
git commit -m "li: V7 Spatial 全量打分"
```

---

## 四、产出物清单

### Phase 2: 统一评测脚本 (`report/unified_eval.py`)

待编写。输入四个打分文件 + `all_data.parquet` + CSI300，输出：
- `tier1_signal.csv` — IC/ICIR/IC>0/月IC
- `tier2_sweep.csv` — N×K 扫参
- `tier3_risk.csv` — Sharpe/MaxDD/月收益
- `tier4_cost.csv` — 成本损耗
- `daily_ic_timeseries.csv` — 日IC序列
- `daily_portfolio.csv` — 日收益序列

### Phase 3: 可视化

**已有图表（17张）**：`report/figures/fig1-17.png`  
生成脚本：`report/plot_charts.py`

**新补图表（~17张）**：
- 信号质量：日IC叠加折线、滚动20日IC、月IC箱线、模型秩相关热力图
- 策略表现：累计收益+回撤阴影、N×K四拼热力图、成本瀑布图、换手率对比
- 截面分析：IC×市值分位、分数分布密度、行业分布、持仓位贡献
- 核心汇总：IC vs 净收益散点、雷达图、Jan vs Feb-May拆解

### Phase 4: 报告整合

更新 `report/实验报告_初稿.md`：
- 替换第五章统一评测数据
- 插入新图表
- 补充 V7 竞赛附录
- 更新附录B图表索引

---

## 五、Phase 1 期间修复的关键 Bug

| Bug | 根因 | 修复方式 |
|-----|------|------|
| 特征构建 O(N²) | `df[df["ts_code"]==ts]` 全量扫描 | 改为 `df.groupby("ts_code")` |
| 模型参数错位 | 原始类含 `bidirectional` 参数，位置传参→`bidirectional=32` | 改为关键字参数 |
| 首batch全NaN | Spatial attention CUDA kernel 未初始化 | 加 warmup forward pass |
| PE缺失→rank_pct NaN | 缺PE股票返回NaN→污染特征张量 | rank_pct默认返回0.5（中性排名） |
| import `F` 误写 | `import F` 应为 `import torch.nn.functional as F` | 修正导入语句 |
| inlined类命名不匹配 | checkpoint用`query/key/value`，inlined类用`q/k/v` | V7/V8改用原始import |

---

## 六、关键脚本位置

| 脚本 | 路径 | 功能 |
|------|------|------|
| 统一打分 | `report/score_models.py` | 批量生成各模型全量打分 |
| V7专用 | `report/score_v7.py` | V7 Spatial打分（备选） |
| 图表生成 | `report/plot_charts.py` | 已有17张图 |
| 图表目录 | `report/figures/` | PNG输出 |
| 报告初稿 | `report/实验报告_初稿.md` | Markdown报告 |
| 统一评测 | `report/unified_eval.py` | **待编写**（Phase 2） |
