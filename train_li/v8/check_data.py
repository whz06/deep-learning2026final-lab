import torch, os, glob
p = "D:/Workspace/DL_HW/deep-learning2026final-lab/train_li/processed/v8_windows"
train = len(glob.glob(f"{p}/train/**/*.pt", recursive=True))
val = len(glob.glob(f"{p}/val/**/*.pt", recursive=True))
print(f"Train files: {train}, Val files: {val}, Total: {train+val}")

all_pt = glob.glob(f"{p}/train/**/*.pt", recursive=True) + glob.glob(f"{p}/val/**/*.pt", recursive=True)
d0 = torch.load(all_pt[0], map_location="cpu", weights_only=True)
d1 = torch.load(all_pt[-1], map_location="cpu", weights_only=True)
print(f"Industry range: {d0['industry_id'].min().item()} - {d0['industry_id'].max().item()}")
print(f"Feature shape: {d0['features'].shape}")
print(f"Feature dtype: {d0['features'].dtype}")
print(f"Labels dtype: {d0['labels'].dtype}")
