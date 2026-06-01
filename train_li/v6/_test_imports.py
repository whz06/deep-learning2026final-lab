import sys, os
sys.path.insert(0, r"D:\Workspace\DL_HW\deep-learning2026final-lab\train_li\v2")
from train import listmle_loss, rankic
from models.gru import GRURanker
print("v2 imports OK")

import importlib.util
_spec = importlib.util.spec_from_file_location("v5_dataset", 
    r"D:\Workspace\DL_HW\deep-learning2026final-lab\train_li\v5\v5_dataset.py")
mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(mod)
print("v5_dataset OK")

sys.path.insert(0, r"D:\Workspace\DL_HW\deep-learning2026final-lab\train_li")
from v6.models.gru_attn import GRURankerAttn
print("gru_attn OK")
from v6.models.spatial_attn import SparseSpatialAttention
print("spatial_attn OK")
from v6.models.gru_spatial import GRURankerSpatial
print("gru_spatial OK")

model = GRURankerAttn(26, 128, 1, 0.2)
print(f"GRURankerAttn params: {sum(p.numel() for p in model.parameters()):,}")

model2 = GRURankerSpatial(26, 128, 1, 0.2, K=10)
print(f"GRURankerSpatial params: {sum(p.numel() for p in model2.parameters()):,}")

print("ALL IMPORTS OK")
