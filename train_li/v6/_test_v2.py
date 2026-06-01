import sys, os
sys.path.insert(0, r"D:\Workspace\DL_HW\deep-learning2026final-lab\train_li\v2")
from train import listmle_loss, rankic
from models.gru import GRURanker
print("v2 imports OK")

sys.path.insert(0, r"D:\Workspace\DL_HW\deep-learning2026final-lab\train_li")
from v6.models.spatial_attn import SparseSpatialAttention
sa = SparseSpatialAttention(d_model=128, d_proj=32, K=10)
print(f"spatial_attn OK: params={sum(p.numel() for p in sa.parameters()):,}")

from v6.models.gru_spatial_v2 import GRURankerSpatialConcat, GRURankerSpatialGated, GRURankerSpatialRes
m1 = GRURankerSpatialConcat(26, 128, 1, 0.2, d_proj=32, K=10)
m2 = GRURankerSpatialGated(26, 128, 1, 0.2, d_proj=32, K=10)
m3 = GRURankerSpatialRes(26, 128, 1, 0.2, d_proj=32, K=10)
print(f"S1 (concat)  params: {sum(p.numel() for p in m1.parameters()):,}")
print(f"S2 (gated)   params: {sum(p.numel() for p in m2.parameters()):,}")
print(f"S3 (residual) params: {sum(p.numel() for p in m3.parameters()):,}")

# Test forward pass
x = torch.randn(512, 60, 26)
print(f"\nS1 forward: {m1(x).shape}")
print(f"S2 forward: {m2(x).shape}")
print(f"S3 forward: {m3(x).shape}")
print("ALL OK")
