"""v5/v5_dataset.py — Dataset loading from processed/v5_windows/ (26-dim features)."""
import os, glob
import torch
from torch.utils.data import Dataset

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_WINDOWS_DIR = os.path.join(_ROOT, "processed", "v5_windows")

_cache_features = {}
_cache_labels = {}


def _preload(split, use_cache=True):
    pattern = os.path.join(_WINDOWS_DIR, split, "**", "*.pt")
    paths = sorted(glob.glob(pattern, recursive=True))
    if not paths:
        raise RuntimeError(f"No .pt files under {os.path.join(_WINDOWS_DIR, split)}")

    if not use_cache:
        return paths

    new = 0
    for path in paths:
        if path not in _cache_features:
            data = torch.load(path, map_location="cpu", weights_only=True)
            _cache_features[path] = data["features"].half()
            _cache_labels[path] = data["labels"]
            new += 1

    if new > 0:
        mb = sum(f.numel() * f.element_size() for f in _cache_features.values()) // 2**20
        print(f"[dataset] Cached {len(paths)} dates from {split} "
              f"(+{new} new, ~{mb} MB total in RAM)")
    return paths


class DailyStockDataset(Dataset):
    def __init__(self, split, window_size=60, sample_size=None, shuffle=True):
        self.split = split
        self.window_size = window_size
        self.sample_size = sample_size
        self.shuffle = shuffle
        self.use_cache = (split == "train")
        self.file_paths = _preload(split, use_cache=self.use_cache)

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        path = self.file_paths[idx]
        if self.use_cache:
            features = _cache_features[path][:, -self.window_size:, :].float()
            labels = _cache_labels[path]
        else:
            data = torch.load(path, map_location="cpu", weights_only=True)
            features = data["features"][:, -self.window_size:, :]
            labels = data["labels"]

        n = len(labels)
        if self.sample_size is not None and n > self.sample_size:
            if self.shuffle:
                perm = torch.randperm(n)[:self.sample_size]
            else:
                perm = torch.arange(self.sample_size)
            features = features[perm]
            labels = labels[perm]

        return features, labels
