# CNN-LSTM 统一评测回测（我的工作目录）

## 文件说明

| 文件 | 说明 |
|:----|:------|
| `test_correct_backtest.py` | 统一评测回测主脚本（加载模型→推理→回测→保存） |
| `test_report_backtest.py` | 前期测试脚本（已废弃） |
| `gen_wu_barchart.py` | 生成 `fig_model_compare_bars.png` 的柱状图脚本 |
| `backtest_n5k3.py` | 更早期的回测脚本（已废弃） |
| `results/backtest_n5k3_daily.csv` | 最新含成本回测结果（N=5 K=3） |
| `results/backtest_n5k3_full_daily.csv` | 全量数据回测结果（已废弃） |
| `results/backtest_n5k3_report.md` | 回测报告 |
| `results/backtest_n5k3_full_report.md` | 全量数据回测报告 |

## 数据依赖

- 模型: `train_wu/CNN-LSTM/result/maxmin_cnn_lstm_L20.pt`
- 数据: `train_wu/CNN-LSTM/maxmin_scale_daily_close.csv`
- CSI300: `train_li/data/market/000300.SH.csv`

## 注意

回测结果存在争议——从 4800 只股票中每日选 5 只极端尾部的测试方式会产生不现实的极高收益。
建议仅以 RankIC=0.067 作为模型评估依据。
