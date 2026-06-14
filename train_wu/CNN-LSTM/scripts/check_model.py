import torch
ckpt = torch.load("result/cnn_lstm_L20.pt", map_location="cpu", weights_only=False)
print("Keys:", list(ckpt.keys()))
if "model_cfg" in ckpt:
    for k, v in ckpt["model_cfg"].items():
        print(f"  {k}: {v}")
if "seq_len" in ckpt:
    print(f"  seq_len (direct): {ckpt['seq_len']}")
