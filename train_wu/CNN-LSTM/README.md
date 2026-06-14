# CNN-LSTM 股票预测

基于 CNN-LSTM 的 A 股下一日收盘价预测 + N-K 旋转回测策略。

## 项目结构

```
CNN-LSTM/
├── train.py                       # 训练入口
├── predict.py                     # 推理入口
├── backtest.py                    # 单独回测入口
├── CLAUDE.md                      # 项目说明
├── README.md
├── maxmin_scale_daily_close.csv   # 渐进式 Min-Max 归一化收盘价
├── data_analysis.md               # 原始数据分析（股票过滤 + 价格分布 + 缩放方法）
├── sigmoid_scaler_plot.png        # Sigmoid 函数曲线
├── sigmoid_vs_distribution.png    # Sigmoid 与 A 股分布错位对比
│
├── src/stock_predictor/           # 核心模块
│   ├── config.py / data.py / models.py
│   ├── trainer.py / metrics.py / strategy.py
│
├── result/
│   ├── experiment_report.md       # 实验报告
│   ├── *.pt                       # 模型权重（详见下方清单）
│   ├── *.json                     # 训练指标
│   └── loss_*.png                 # 训练过程可视化
│
└── scripts/                       # 辅助脚本
```

## 模型文件清单（result/）

| 文件 | 说明 |
|------|------|
| **maxmin_cnn_lstm_L20.pt** | **最佳模型** - 3年混合，CNN-LSTM L20 |
| sigmoid_cnn_lstm_L20.pt | 3年混合 sigmoid（无效） |
| maxmin_cnn_lstm_L60.pt | 3年 L60 扩展实验 |
| sigmoid_cnn_lstm_L60.pt | 3年 L60 sigmoid（无效） |
| single_maxmin_lstm_L10.pt | 单年 LSTM L10（前期） |
| single_maxmin_lstm_L20.pt | 单年 LSTM L20（前期） |
| single_sigmoid_lstm_L10.pt | 单年 sigmoid LSTM |
| single_sigmoid_lstm_L20.pt | 单年 sigmoid LSTM |
| single_sigmoid_cnn_lstm_L10.pt | 单年 sigmoid CNN-LSTM |
| single_sigmoid_cnn_lstm_L20.pt | 单年 sigmoid CNN-LSTM |
| old_cnn_lstm_L5.pt | 早期 L5 实验 |

## 实验结论

详见 `result/experiment_report.md`。最佳方案：**3年混合 + CNN-LSTM L20 + maxmin**。
- IC = 0.118，RankIC = 0.088，方向胜率 52.45%
- 132 天回测累计 +8,500%（1M 到 86M），Sharpe 16.41

## 训练命令

```bash
# 主实验：3年混合（需 data.py year 过滤 >=）
python train.py --model cnn_lstm --scale maxmin --seq_lens 20 --year 2024

# 前期尝试
python train.py --model lstm --scale all --seq_lens 10,20 --year 2026
```

## 预测与回测

```bash
python predict.py --model_path result/maxmin_cnn_lstm_L20.pt --last_days 3 --top_k 10
python backtest.py --signals_csv <信号文件> --n 10 --k 3
```

## 数据说明

- 数据源：A 股 4,945 只股票，2016-01-04 ~ 2026-05-29（9,474,585 行）
- 输入特征：仅收盘价（单变量时间序列）
- 归一化：逐股票渐进式 Min-Max
- 预测目标：下一交易日归一化收盘价 → 预期收益率
