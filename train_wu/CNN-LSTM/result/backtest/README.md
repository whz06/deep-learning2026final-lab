# CNN-LSTM N=5, K=3 统一评测回测

## 文件清单

| 文件 | 说明 |
|:----|:-----|
| `backtest_n5k3_daily.csv` | 逐日回测明细（signal_date, port_ret, n_hold, trade_cost, csi5d, risk_off） |
| `backtest_n5k3_summary.json` | 汇总指标（累积收益、Sharpe、最大回撤等） |
| `backtest_n5k3_report.md` | 完整回测报告（含对比分析、结论） |
| `README.md` | 本文件：生成过程说明 |

## 生成过程

**生成脚本不在本目录**，因为需要引用 `src/stock_predictor/` 下的模块。

### 脚本位置

```
train_wu/CNN-LSTM/
├── backtest_n5k3.py          ← 回测入口脚本
├── src/stock_predictor/
│   ├── data.py               ← 数据加载 + 滑动窗口构建
│   ├── trainer.py            ← 模型加载 + predict_arrays()
│   ├── models.py             ← 模型定义
│   ├── config.py             ← 配置
│   └── ...
└── result/
    ├── maxmin_cnn_lstm_L20.pt   ← 模型权重
    └── backtest/                ← 回测结果输出目录
```

### 运行方式

```bash
cd train_wu/CNN-LSTM
python backtest_n5k3.py
```

### 执行流程

```
backtest_n5k3.py
  │
  ├── load_checkpoint("result/maxmin_cnn_lstm_L20.pt")
  │   └── 加载模型权重 + 恢复训练配置（model, seq_len, scale_name...）
  │
  ├── load_scale_frame(scale_name="maxmin", year=2024)
  │   └── 读取 maxmin_scale_daily_close.csv（23 万行，5518 只股票）
  │
  ├── build_splits(df, seq_len=20)
  │   └── 按股票分组 → 向量化滑动窗口 → train/val/test 切分
  │
  ├── predict_arrays(model, test_arr, batch_size=4096)
  │   └── 模型推理 → inverse_maxmin 反归一化 → 计算 pred_ret
  │
  └── run_nk_backtest(signals, csi5d_map)
      ├── 按 signal_date 分组，取 2026-01-05 ~ 2026-05-29
      ├── 每日排序 → 买卖决策（N=5, K=3）
      ├── 计算等权收益 + 净值法交易成本
      └── 保存 backtest_n5k3_daily.csv + backtest_n5k3_summary.json
```

### 数据依赖

- `maxmin_scale_daily_close.csv` — 缩放后的日线收盘价（含训练/测试标记）
- `train_li/data/market/000300.SH.csv` — CSI300 指数数据（用于风控判断）

### 注意事项

1. **生成脚本不在此目录**：`backtest_n5k3.py` 位于 `train_wu/CNN-LSTM/` 根目录，因为需要引用 `src/` 下的模块。此目录仅存放运行结果。
2. **运行环境**：需在 `train_wu/CNN-LSTM/` 根目录执行，确保 Python import 路径正确。
3. **模型依赖**：回测基于已训练好的 `maxmin_cnn_lstm_L20.pt` 模型，如果重新训练需重新运行回测。
