# 移动与 Git 协作计划

> 本文档供新对话读取恢复上下文。已完成移动操作，代码在 `train_li/` 目录下。

---

## 一、目标目录结构（当前状态）

```
deep-learning2026final-lab/              ← Git 仓库根目录（github.com/whz06/deep-learning2026final-lab）
├── .gitignore                          ← 仅 IDE/OS 通用规则（4行极简版）
├── README.md                           ← 仓库总说明（已完成）
│
├── train_li/                           ← li 同学的全部项目（原 LAB5）
│   ├── .gitignore                      ← 只控制 train_li/ 下的忽略规则
│   ├── shared/preprocess.py            ← 数据预处理
│   ├── trade/infer.py                  ← 比赛日推理
│   ├── v1/ v2/ v3/ v4/ v5/           ← 实验版本
│   ├── sharedcontext/                  ← 项目文档（10个md）
│   ├── data/          (gitignored)
│   └── processed/     (gitignored)
│
├── train_wu/                           ← wu 同学的项目（完全独立）
│   ├── .gitignore
│   └── README.md
│
└── train_wang/                         ← wang 同学的项目（完全独立）
    ├── .gitignore
    └── README.md
```

## 二、移动操作（已完成 ✅）

原 `/d/Workspace/DL_HW/LAB5/` 已移动至 `train_li/`（即原计划中的 `li/`）。实际团队成员为 li、wu、wang，对应文件夹 `train_li`、`train_wu`、`train_wang`。

```bash
# 已执行
mkdir deep-learning2026final-lab
mv LAB5 deep-learning2026final-lab/train_li
cd deep-learning2026final-lab
git init
```

所有脚本用 `os.path.dirname(os.path.dirname(os.path.abspath(__file__)))` 反推根目录，移动后自动适配 `train_li/`。

## 三、Git 设置（已完成 ✅）

### 根 `.gitignore`（极简，三个人不改）

```
.idea/
.vscode/
.DS_Store
Thumbs.db
```

### `train_li/.gitignore`（只管自己的数据）

```
data/daily/
data/metric/
data/moneyflow/
data/index_weight/
processed/
*/checkpoints/*.pt
__pycache__/
*.pyc
trade/buy_list.txt
trade/sell_list.txt
trade/decision.log
*/results/*.parquet
*/results/*.csv
*.log
```

### 进 git 的文件（`git add train_li/` 后自动包含）

- `train_li/**/*.py` — Python 源码
- `train_li/**/*.ps1` — PowerShell 脚本
- `train_li/data/basic.csv`、`train_li/data/trade_cal.csv`、`train_li/data/market/*.csv` — 小文件 (<2MB)
- `train_li/sharedcontext/*.md` — 项目文档
- `train_li/trade/README.md`

### 不进 git 的（被 .gitignore 排除）

- `train_li/data/daily/` (~1GB)
- `train_li/data/metric/` (~1.5GB)
- `train_li/data/moneyflow/` (~1.4GB)
- `train_li/processed/` (v2_windows 26GB + v5_windows ~700MB + all_data.parquet)
- `train_li/v2/checkpoints/` (273MB)
- `train_li/v5/checkpoints/` (272KB — 会被加进 git 因为文件小，但建议 exclude)
- `train_li/trade/buy_list.txt` `sell_list.txt` `decision.log`

## 四、协作规则

1. 每人只在**自己命名**的子文件夹（`train_li/` / `train_wu/` / `train_wang/`）中新建/修改代码
2. 自己的 `.gitignore` 只控制自己的目录 —— 改别人的 `.gitignore` 前要打招呼
3. `git add train_li/` 只加自己的文件夹，不要 `git add .`
4. 原始数据（`data/daily/` 等）各自从课程渠道获取，各放各的文件夹
5. 最后写报告时，从三个人的文件夹中各自抽取结论

## 五、当前 v5 训练状态（已更新）

### 训练已完成 ✅

v5 已训练完成，结果优于 v2：

| 项目 | 内容 |
|------|------|
| best val IC (v5.1) | **0.1030** @ epoch 27 (T+1, 无预处理, 26-dim fix) |
| best val IC (v5.2) | **0.1114** @ epoch 9 (T+5, log1p+winsor+PE/PB clip) |
| v2 best val IC | 0.1029 (GRU H=128 L=1 D=0.2) |
| v5 vs v2 improvement | **+8.2%** (0.1114 vs 0.1029) |
| 特征 | 26 维 (19 temporal + 7 cross/valuation) |
| checkpoint | `v5/checkpoints/gru_v5_H128_L1_D0.2_lr0.0003_N2048.pt` |
| 训练时间 | 18 min (AMP+compile+val_every=3)

## 六、sharedcontext 文档索引

移动后 `sharedcontext/` 有 9 个文件：

| 文件 | 内容 |
|------|------|
| `PROJECT.md` | 项目结构、数据流、路径映射 |
| `EXPERIMENTS.md` | V1→V4 所有实验记录 |
| `KEY_DECISIONS.md` | 关键设计决策 + 失败方案汇总 |
| `PLAN_CRITIQUE.md` | 对《初步计划.md》的批判性分析（含 KNN proxy 验证结果） |
| `SPATIAL_ATTENTION_PLAN.md` | 旧空间注意力 Phase 1-4 计划（已归档） |
| `plan0601.md` | 新五步优化计划：Step 1-5 |
| `初步计划.md` | 原根目录的初步设计方案（含双阶段注意力） |
| `作业要求.md` | 课程作业要求 |
| `实验环境.md` | 硬件/环境配置说明 |

## 七、新对话首次读取

在新对话中，让大模型工具先读这个文件：

```
请先读取 sharedcontext/PROJECT.md 了解项目结构，然后读取 TRANSFER_PLAN.md 了解当前进度。
```

## 八、待完成任务（更新后）

- [x] 移动完成 → `train_li/`
- [x] v5 训练完成 (best IC=0.1114 > v2 0.1029)
- [x] 根 `.gitignore` 替换为极简版
- [x] 创建 `train_li/.gitignore`
- [x] 创建 `train_wu/`、`train_wang/` 脚手架
- [x] 重写根 `README.md`
- [ ] 验证 `trade/infer.py` 用 v5 checkpoint 推理效果
- [ ] 修复 `trade/infer.py` 预处理（缺少 log1p、winsorization、PE/PB clip）
- [ ] 如果 v5 推理验证通过，删除 `processed/v2_windows/`（~26GB）
- [ ] push 到 GitHub
- [ ] 通知 wu 和 wang 克隆仓库并放入自己的代码
