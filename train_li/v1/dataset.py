"""
v1/dataset.py — PyTorch Dataset, pre-loads all .pt windows into RAM at init.

Each __getitem__ returns (features [N,T,F], labels [N]) for one date from cache.
No disk I/O, no pickle after startup.
"""
import os
import glob
import torch
from torch.utils.data import Dataset


class DailyStockDataset(Dataset):
    def __init__(self, window_dir: str, window_size: int = 60,
                 sample_size: int = None, shuffle: bool = True):
        self.window_dir = window_dir
        self.window_size = window_size
        self.sample_size = sample_size
        self.shuffle = shuffle

        self.file_paths = sorted(
            glob.glob(os.path.join(window_dir, "**", "*.pt"), recursive=True)
        )
        if len(self.file_paths) == 0:
            raise RuntimeError(f"No .pt files found under {window_dir}")

        print(f"[dataset] Pre-loading {len(self.file_paths)} date files into RAM ...")
        self._features_cache = []
        self._labels_cache = []

        for i, path in enumerate(self.file_paths):
            data = torch.load(path, map_location="cpu", weights_only=True)
            # Store features as float16 to halve RAM footprint (7 GB instead of 14 GB)
            self._features_cache.append(data["features"].half())
            self._labels_cache.append(data["labels"])  # float32, tiny
            if (i + 1) % 300 == 0:
                print(f"  ... {i + 1}/{len(self.file_paths)}")

        print(f"[dataset] Cached {len(self.file_paths)} dates "
              f"(~{sum(f.numel() * f.element_size() for f in self._features_cache) // 2**20} MB)")

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        features = self._features_cache[idx].float()   # float16→float32
        labels = self._labels_cache[idx]                # float32

        # Slice to desired window_size
        features = features[:, -self.window_size:, :]

        n = len(labels)
        if self.sample_size is not None and n > self.sample_size:
            if self.shuffle:
                perm = torch.randperm(n)[:self.sample_size]
            else:
                perm = torch.arange(self.sample_size)
            features = features[perm]
            labels = labels[perm]

        return features, labels
