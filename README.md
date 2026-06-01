# deep-learning2026final-lab — 过拟合散户小组

> 基于深度学习的 A 股股票排序预测与模拟交易  
> 课程：深度学习2026大作业

---

## 仓库结构

```
deep-learning2026final-lab/
├── .gitignore              ← 仅忽略 IDE 配置和系统垃圾，三人共享，不要改
├── README.md               ← 本文件
├── LICENSE                 ← MIT
│
├── train_li/               ← li 同学的项目（GRU 股票排序 + 策略回测）
│   ├── .gitignore          ← 只控制 train_li/ 下的忽略规则
│   ├── sharedcontext/      ← 项目文档（PROJECT.md、实验记录、环境说明）
│   ├── trade/              ← 比赛日推理（买入/卖出列表生成）
│   ├── v1/ v2/ v3/ v4/ v5/← 实验版本迭代
│   └── ...
│
├── train_wu/               ← wu 同学的项目
│   ├── .gitignore          ← 只控制 train_wu/ 下的忽略规则
│   └── README.md           ← 接入指南
│
└── train_wang/             ← wang 同学的项目
    ├── .gitignore          ← 只控制 train_wang/ 下的忽略规则
    └── README.md           ← 接入指南
```

---

## 协作规则（三人必须遵守）

| # | 规则 | 说明 |
|---|------|------|
| 1 | **只改自己的文件夹** | `train_li/`、`train_wu/`、`train_wang/` 各自独立开发 |
| 2 | **`git add train_xx/`** | 提交时显式指定自己的文件夹，**禁止 `git add .`** |
| 3 | **`.gitignore` 各管各的** | 每个人的 `.gitignore` 在各自子文件夹内，互不影响 |
| 4 | **根 `.gitignore` 不要改** | 只排除了 `.idea/`、`.vscode/`、`.DS_Store`、`Thumbs.db` |
| 5 | **大文件不提交** | 原始数据（`data/daily/` 等）、checkpoint（`.pt`）、处理后的中间文件各自通过 `.gitignore` 排除 |
| 6 | **原始数据各自获取** | 课程提供的数据文件各人自己下载，不通过 Git 共享 |
| 7 | **先 pull 再 push** | 推之前拉取他人更新，避免冲突 |
| 8 | **不要 force push** | 永远不要 `--force` 推送到 main |

---

## 首次使用（新成员）

1. **克隆仓库**

   ```bash
   git clone git@github.com:whz06/deep-learning2026final-lab.git
   ```

2. **把自己的代码放进对应文件夹**

   详细指南见你自己文件夹中的 `README.md`（例如 `train_wang/README.md`）。有两种方式：
   - **方式一（推荐）**：把整个项目文件夹移入，然后新开 AI 对话开始开发
   - **方式二**：在原目录继续开发，每次提交时复制源码文件过来

3. **配置 `.gitignore`**

   根据你的项目类型，在子文件夹的 `.gitignore` 中添加需要排除的文件规则。

4. **开始开发 + 提交**

   ```bash
   git add train_你的名字/
   git commit -m "你的名字: 干了什么"
   git push
   ```

---

## 当前进度

| 成员 | 模型 | 最佳 Val IC | 策略 | 状态 |
|------|------|------------|------|------|
| li | GRU (H=128, L=1, 26-dim) | 0.1114 | Strategy B (动量止损) | v5 训练完成 |
| wu | — | — | — | 待接入 |
| wang | — | — | — | 待接入 |

---

## 环境信息

- **GPU**: RTX 4060 Laptop 8GB VRAM
- **Python**: conda `dl_lab1`, PyTorch 2.6.0+cu124
- **OS**: Windows + WSL2
- **训练**: Windows PowerShell 执行 `.ps1` 脚本
- 详见 `train_li/sharedcontext/实验环境.md`
