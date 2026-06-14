import sys, os, torch
ROOT = r"D:\Workspace\DL_HW\deep-learning2026final-lab\train_li"
sys.path.insert(0, ROOT)
from v7.models.gru_spatial import GRURankerSpatial
from v7.models.spatial_attn import SparseSpatialAttention

m = GRURankerSpatial(input_dim=26, hidden_size=128, num_layers=1, dropout=0.2, d_proj=32, K=5)
print(f"Class: {type(m).__name__}, Params: {sum(p.numel() for p in m.parameters()):,}")

# Load checkpoint
ckpt = os.path.join(ROOT, "v7", "checkpoints", "gru_spatial_v7_d32_K5_H128_L1_D0.2_lr0.0003_N1024.pt")
state = torch.load(ckpt, map_location="cpu", weights_only=True)
state = {k.replace("_orig_mod.", ""): v for k, v in state.items()}
m.load_state_dict(state, strict=True)
print("State dict loaded OK (strict=True)")

# Test forward pass with random input
x = torch.randn(100, 60, 26)
m.eval()
with torch.no_grad():
    y = m(x)
print(f"Forward pass: shape={y.shape}, NaN={torch.isnan(y).sum().item()}, mean={y.mean().item():.4f}")
