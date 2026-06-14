import pandas as pd
sig = pd.read_csv("result/cnn_lstm_L20_test_signals.csv.gz")
jun5 = sig[sig["signal_date"] == 20260605].sort_values("pred_ret", ascending=False).head(3)
print("=== 6月5日信号日 -> 6月8日目标日 Top3 ===")
for _, r in jun5.iterrows():
    print(f"  {r['ts_code']}: 预测={r['pred_ret']*100:.2f}%, 实际={r['real_ret']*100:.2f}%, 买入={r['cur_close']:.2f}, 卖出={r['next_close']:.2f}")
print(f"\nTop3 等权实际收益: {jun5['real_ret'].mean()*100:.2f}%")
