# train_wang — 深度学习2026大作业

> 这是 `wang` 同学的工作目录。所有的 `.py` 源代码、实验脚本、推理逻辑都放在这个文件夹里。

---

## 快速开始：如何把你的代码接入这个仓库

你的代码目前可能在另一个文件夹里（比如 `D:\Workspace\DL_HW\MY_PROJECT\`），有两种方式接入这个 Git 仓库。

### 方式一：整体移动到子文件夹 + 新建对话开发（推荐）

适用于：你打算后续全部在这个仓库目录下开发，旧目录不再使用。

1. **把整个项目文件夹移动过来**

   ```bash
   mv /d/Workspace/DL_HW/MY_PROJECT /d/Workspace/DL_HW/deep-learning2026final-lab/train_wang/
   ```

   或者在 Windows PowerShell 中：

   ```powershell
   Move-Item D:\Workspace\DL_HW\MY_PROJECT D:\Workspace\DL_HW\deep-learning2026final-lab\train_wang\
   ```

2. **配置 `.gitignore`**

   你的 `train_wang/.gitignore` 目前有一个基础模板。请根据自己的项目补充需要忽略的内容（大体积数据、checkpoint、日志文件等）。

   **重要：** `.gitignore` 只控制 `train_wang/` 目录下的文件，不会影响 `train_li/` 和 `train_wu/` 里的文件。

   推荐至少添加：

   ```gitignore
   data/daily/
   data/metric/
   processed/
   */checkpoints/*.pt
   ```

3. **新开一个对话，让 LLM 恢复上下文**

   在新对话中告诉 LLM：

   ```
   请先读取 train_wang/README.md 了解项目目标和当前进度
   ```

   建议在 `train_wang/` 下写一个简要的文件说明你的实验进展，方便 AI 在新对话中快速理解上下文。

4. **开始开发**

   之后所有的代码修改都在 `train_wang/` 内进行。Git 提交时**只加自己的文件夹**：

   ```bash
   git add train_wang/
   git commit -m "wang: 修改了xxx"
   ```

   **绝对不要用 `git add .`**——这会误加其他人的文件。

### 方式二：在原目录开发 + 复制代码文件到子文件夹提交

适用于：你习惯在原来的开发目录工作，不想移动文件，只把需要共享的代码文件定时同步过来。

1. **在原目录正常开发**（比如 `D:\Workspace\DL_HW\MY_PROJECT\`）

2. **每次需要提交时，把源码复制过来**

   ```bash
   # 只复制 .py 源码文件（数据、checkpoint 不要复制）
   cp -r /d/Workspace/DL_HW/MY_PROJECT/*.py /d/Workspace/DL_HW/deep-learning2026final-lab/train_wang/
   cp -r /d/Workspace/DL_HW/MY_PROJECT/subdir/ /d/Workspace/DL_HW/deep-learning2026final-lab/train_wang/
   ```

   或者在 Windows PowerShell 中：

   ```powershell
   Copy-Item D:\Workspace\DL_HW\MY_PROJECT\*.py D:\Workspace\DL_HW\deep-learning2026final-lab\train_wang\
   Copy-Item D:\Workspace\DL_HW\MY_PROJECT\subdir\ D:\Workspace\DL_HW\deep-learning2026final-lab\train_wang\ -Recurse
   ```

3. **提交**

   ```bash
   git add train_wang/
   git commit -m "wang: 同步最新代码"
   ```

4. **善用 `.gitignore`**

   如果原目录生成了新类型的临时文件，记得同步更新 `train_wang/.gitignore`，避免把大文件或中间产物提交到仓库。

---

## Git 协作规则

| 规则 | 说明 |
|------|------|
| 每人只改自己的文件夹 | `train_li/`、`train_wu/`、`train_wang/` 各自独立 |
| 用 `git add train_wang/` 提交 | **禁止** `git add .`，只加自己的 |
| `.gitignore` 只管自己 | 每个人的 `.gitignore` 在自己子文件夹内，互不影响 |
| 根 `.gitignore` 不要改 | 根目录的 `.gitignore` 只阻止 IDE 配置文件和系统垃圾，三人共享 |
| 原始数据各自获取 | `data/daily/`、`data/metric/` 等大文件不提交，各自从课程渠道下载 |

---

## 文件结构建议

```
train_wang/
├── .gitignore          ← 控制 train_wang/ 下哪些文件不进 git
├── README.md           ← 本文件（上下文说明）
├── sharedcontext/      ← （可选）放你自己的项目文档
├── data/               ← 原始数据（被 .gitignore 排除）
├── processed/          ← 处理后的数据（被 .gitignore 排除）
├── models/             ← 模型定义
├── train.py            ← 训练脚本
├── infer.py            ← 推理脚本
└── ...
```

---

## 环境说明

- **GPU**: RTX 4060 Laptop 8GB VRAM
- **Python**: conda 环境 `dl_lab1`，PyTorch 2.6.0+cu124
- **操作系统**: Windows + WSL2
- **训练执行**: 在 Windows PowerShell 中直接运行 `.ps1` 脚本或调用 conda python

如有环境配置问题，参考 `train_li/sharedcontext/实验环境.md`。
