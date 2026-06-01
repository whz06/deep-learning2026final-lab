# train_wu — 深度学习2026大作业

> 这是 `wu` 同学的工作目录。所有的 `.py` 源代码、实验脚本、推理逻辑都放在这个文件夹里。

---

## 快速开始：如何把你的代码接入这个仓库

你的代码目前可能在另一个文件夹里，有两种方式接入这个 Git 仓库。

### 方式一：整体移动到子文件夹 + 新建对话开发（推荐）

适用于：你打算后续全部在这个仓库目录下开发，旧目录不再使用。

1. **把整个项目文件夹移动过来**

   ```bash
   mv /你的/原始/项目/路径 /仓库根目录/train_wu/
   ```

   或者在 Windows PowerShell 中：

   ```powershell
   Move-Item "你的原始项目路径" "仓库根目录\train_wu\"
   ```

2. **检查 `.gitignore` 是否正确**

   你的项目里应该已经有一个 `train_wu/.gitignore` 文件。它继承了一份通用的 Python `.gitignore` 模板。

   如果你有额外的需要忽略的文件（比如大体积数据、checkpoint、日志），请自己补充规则到 `train_wu/.gitignore` 中。

   **重要：** `.gitignore` 只控制 `train_wu/` 目录下的文件，不会影响 `train_li/` 和 `train_wang/` 里的文件。

3. **新开一个对话，让 LLM 恢复上下文**

   在新对话中告诉 LLM：

   ```
   请先读取 train_wu/README.md 了解项目目标和当前进度。
   ```

   建议在 `train_wu/` 下写一个简要的文件说明你的实验进展（可以就叫 `PLAN.md` 或放在 `sharedcontext/` 里），方便 AI 在新对话中快速理解上下文。

4. **开始开发**

   之后所有的代码修改都在 `train_wu/` 内进行。Git 提交时**只加自己的文件夹**：

   ```bash
   git add train_wu/
   git commit -m "wu: 修改了xxx"
   ```

   **绝对不要用 `git add .`**——这会误加其他人的文件。

### 方式二：在原目录开发 + 复制代码文件到子文件夹提交

适用于：你习惯在原来的开发目录工作，不想移动文件，只把需要共享的代码文件定时同步过来。

1. **在原目录正常开发**

2. **每次需要提交时，把源码复制过来**

   ```bash
   # 只复制 .py 源码文件（数据、checkpoint 不要复制）
   cp -r /你的/原始/项目/*.py /仓库根目录/train_wu/
   cp -r /你的/原始/项目/子目录/ /仓库根目录/train_wu/
   ```

   或者在 Windows PowerShell 中：

   ```powershell
   Copy-Item "你的原始项目\*.py" "仓库根目录\train_wu\"
   Copy-Item "你的原始项目\子目录" "仓库根目录\train_wu\" -Recurse
   ```

3. **提交**

   ```bash
   git add train_wu/
   git commit -m "wu: 同步最新代码"
   ```

4. **善用 `.gitignore`**

   注意 `train_wu/.gitignore` 已经排除了很多不需要提交的文件（`__pycache__/`、`*.pyc` 等）。如果你在原目录生成了新类型的临时文件，记得同步更新 `train_wu/.gitignore`。

---

## Git 协作规则

| 规则 | 说明 |
|------|------|
| 每人只改自己的文件夹 | `train_li/`、`train_wu/`、`train_wang/` 各自独立 |
| 用 `git add train_wu/` 提交 | **禁止** `git add .`，只加自己的 |
| `.gitignore` 只管自己 | 每个人的 `.gitignore` 在自己子文件夹内，互不影响 |
| 根 `.gitignore` 不要改 | 根目录的 `.gitignore` 只阻止 IDE 配置和系统垃圾，三人共享 |
| 原始数据各自获取 | `data/daily/`、`data/metric/` 等大文件不提交，各自从课程渠道下载 |

---

## 文件结构建议

```
train_wu/
├── .gitignore          ← 控制 train_wu/ 下哪些文件不进 git
├── README.md           ← 本文件（上下文说明）
├── sharedcontext/      ← （可选）放你自己的项目文档
├── data/               ← 原始数据（被 .gitignore 排除）
├── processed/          ← 处理后的数据（被 .gitignore 排除）
├── models/             ← 模型定义
├── train.py            ← 训练脚本
├── infer.py            ← 推理脚本
└── ...
```
