# GN-CNN-LSTM 精简版说明

## 目录结构
- `src/stock_predictor/`：核心模块
- `train.py`：训练入口，训练结束后自动输出测试指标与回测结果
- `predict.py`：用训练好的模型生成最近几天的 TopK 建仓建议
- `backtest.py`：对已有测试信号做 N-K 回测
- `legacy/`：旧版脚本备份
- `result/`：模型、指标、测试信号、回测结果

## 数据说明
- `maxmin_scale_daily_close.csv`：包含 `close_raw`、`min_ref`、`max_ref`、`close_scaled`、`train`
- `sigmoid_scale_daily_close.csv`：包含 `scaled_close`
- 当前实现把两份数据按 `ts_code + trade_date` 合并后使用：
  - 输入特征：`maxmin_scaled`、`sigmoid_scaled`
  - 预测目标：下一时点的 `maxmin_scaled`
  - 反归一化：使用 `min_ref/max_ref` 恢复到原始价格

## 训练命令
```bash
python train.py --model lstm --scale all --seq_lens 5,10,20
```

训练结束后会输出：
- 缩放后测试指标：按连续 `12` 个交易日分段后汇总的 `MSE / MAE / RMSE / R2`
- 原始价格测试指标：按连续 `12` 个交易日分段后汇总的 `MSE / MAE / RMSE / R2`
- `IC / IR / RankIC / RankIR / 方向胜率`：按连续 `12` 个交易日分段计算
- 缺失值处理：评估前会检查缺失，若存在缺失则用对应列均值填充
- 回测指标：`年化收益率 / 最大回撤 / 夏普比率`

## 最新建仓建议
```bash
python predict.py --model_path "result\\lstm_L20.pt" --last_days 3 --top_k 10
```

输出：
- `result/*_recent_signals.csv.gz`
- `result/*_recent_top10.md`

## 单独回测
```bash
python backtest.py --signals_csv "result\\lstm_L20_test_signals.csv.gz" --n 10 --k 3
```

## 保留的可传入参数
- `train.py`
  - `--model`：`lstm / cnn / cnn_lstm`
  - `--scale`：`maxmin / sigmoid / all`
  - `--seq_lens`：窗口长度列表
  - `--device`：`auto / cpu / cuda`
- `predict.py`
  - `--model_path`：训练好的模型路径
  - `--last_days`：最近几个信号日
  - `--top_k`：每个信号日输出多少只股票
  - `--device`：推理设备
- `backtest.py`
  - `--signals_csv`：测试信号文件
  - `--n`：持仓股票数
  - `--k`：每日最大换仓数
