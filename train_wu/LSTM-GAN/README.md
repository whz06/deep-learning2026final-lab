# LSTM-GAN — A-share Stock Return Prediction

基于 WGAN-GP 的 A 股收益预测模型。生成器 G 与判别器 D 均为 LSTM 结构。

## 项目结构

```
LSTM-GAN/
├── src/factor_gan_lab/       # 主包
│   ├── __init__.py
│   ├── data/
│   │   ├── __init__.py
│   │   └── dataset.py        # 数据加载、序列构建、prebuilt .pt 加载
│   ├── models/
│   │   ├── __init__.py
│   │   └── factor_gan.py     # GeneratorLSTM, DiscriminatorLSTM, FactorGAN, FactorGANConfig
│   └── training/
│       ├── __init__.py
│       └── engine.py         # WGAN-GP 训练引擎、滚动窗口、预测
├── scripts/
│   ├── train_factor_gan.py   # 训练入口
│   ├── factor_gan_smoke.py   # 前向验证
│   └── check_missing_values.py
├── shared/
│   ├── preprocess.py         # 原始 CSV → Parquet 合并
│   ├── build_windows_v7.py   # 26 维因子窗口构建
│   ├── build_windows.py
│   └── build_windows_v5.py
├── train.sh                  # SLURM 训练脚本
├── pyproject.toml
└── README.md
```

## 模型架构

### Generator G

```
输入: z (B, T, 26)       ← T=60, F=26
       ↓
LSTM(hidden=256, layers=2, batch_first=True)
       ↓ h_last (B, 256)
Linear(256→64) → LeakyReLU(0.2) → Linear(64→1)
       ↓
输出: r_hat (B,)          ← 预测收益率
```

### Discriminator D

```
输入: z (B, T, 26), returns (B,)
       ↓
LSTM(hidden=128, layers=2, batch_first=True)
       ↓ h_last (B, 128)
concat([h_last, returns]) → (B, 129)
       ↓
Dropout → Linear(129→256) → LeakyReLU → Linear(256→128) → LeakyReLU → Linear(128→1)
       ↓
输出: score (B,)           ← WGAN 实数分数（不接 sigmoid）
```

### 损失函数

```
D_loss = -mean(D(real)) + mean(D(fake)) + gradient_penalty
G_loss = -mean(D(fake)) + mse_weight × MSE(r_hat, y)
```

## 数据流

```
原始日频 CSVs
    ↓ [shared/preprocess.py]
all_data.parquet
    ↓ [shared/build_windows_v7.py]
processed/windows_v7/{train,val}/{year}/{date}.pt
    ├── features: (N, 60, 26)    ← 26 维因子
    ├── labels:   (N,)            ← T+1 收益率
    └── ts_codes: list[str]
    ↓ [scripts/train_factor_gan.py --data_mode prebuilt]
滚动窗口训练 (WGAN-GP)
    ↓
outputs/experiments/{window_tag}/
    ├── history.json
    ├── model.pt
    └── test_pred.csv
```

## 运行方式

### 方式一：完整流水线

```bash
# Step 1: 合并 daily + metric → parquet
python shared/preprocess.py --data_dir ./documents-export-2026-5-15

# Step 2: 构建 26 维因子窗口
python shared/build_windows_v7.py --parquet processed/all_data.parquet

# Step 3: 训练
python scripts/train_factor_gan.py --data_mode prebuilt --windows_dir processed/windows_v7
```

### 方式二：SLURM 集群

```bash
# 修改 train.sh 中的 PROJECT_DIR 和 WINDOWS_DIR
sbatch train.sh
```

### 方式三：CSV 模式（使用已有特征数据）

```bash
python scripts/train_factor_gan.py --data_mode csv --data your_data.csv
```

### 参数说明

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--train_days` | 504 | 训练集交易日数 |
| `--val_days` | 63 | 验证集交易日数 |
| `--test_days` | 21 | 测试集交易日数 |
| `--step_days` | 21 | 窗口滑动步长 |
| `--max_epochs` | 200 | 最大训练轮数 |
| `--patience` | 20 | 早停耐心值 |
| `--batch_size` | 128 | 批次大小 |
| `--lr` | 3e-5 | 学习率 |
| `--n_critic` | 3 | 每步判别器更新次数 |
| `--mse_weight` | 0.5 | MSE 损失权重 |
| `--max_windows` | 0 | 0=跑全部，1=只跑1个 |

## 特征说明（v7，26 维）

| 特征组 | 维度 | 内容 |
|--------|------|------|
| 原始行情 | 10 | open, high, low, close, vol, amount, pct_chg, turnover_rate, volume_ratio, total_mv |
| 技术指标 | 8 | MACD, MACD_signal, RSI, BB_width, BB_pct, mom_5, mom_20, vol_20 |
| VWAP gap | 1 | close/vwap - 1 |
| 截面排名 | 6 | pct_chg, amount, turnover_rate 百分位排名; PE/PB 截断后排名 |
| rel_beta | 1 | 个股 pct_chg - 指数 pct_chg |

预处理：对数变换(vol, amount, total_mv) → Winsorization[P1,P99] → 时序Z-score → 截面Z-score

## 依赖

```
Python >= 3.10
numpy, pandas, torch, scipy, matplotlib
```

## 训练结果示例

```
Epoch   1 | D_loss=-1.2610 G_loss=4.8062 Adv=0.1418 MSE=9.3289 GP=0.2451 | val_mse=10.1784 val_ic=0.0460 dir_hit=0.508
Epoch   2 | D_loss=-1.6107 G_loss=4.7024 Adv=0.0404 MSE=9.3240 GP=0.2580 | val_mse=10.1688 val_ic=0.0628 dir_hit=0.514
```

## 注意事项

- `build_windows_v7.py` 默认只处理 2019-01-02 ~ 2025-12-31 的数据，如需更多年份请修改脚本中的日期范围
- 预构建的 .pt 文件可使用 CPU 生成（无需 GPU），上传到服务器后直接训练
- 标签 `pct_chg` 为百分比值（如 -0.543），在模型中直接使用
