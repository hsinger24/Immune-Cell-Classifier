"""
Data loading, transforms, and group-aware splitting for the blood cell classifier.

Handles two evaluation modes:
  1. Official split  — uses dataset2's TRAIN/TEST as-is (comparable to baselines).
  2. Grouped split   — groups by sourceID so no augmentation of the same cell
                       appears in both train and val/test (leakage-free).
"""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Literal

import numpy as np
import torch
from PIL import Image
from sklearn.model_selection import GroupShuffleSplit
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import transforms as T

# ── Constants ────────────────────────────────────────────────────────────────

CLASSES = ["EOSINOPHIL", "LYMPHOCYTE", "MONOCYTE", "NEUTROPHIL"]
CLASS_TO_IDX = {c: i for i, c in enumerate(CLASSES)}
NUM_CLASSES = len(CLASSES)

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

DEFAULT_DATA_ROOT = Path(__file__).resolve().parent.parent / "dataset2-master" / "dataset2-master" / "images"


# ── Transforms ───────────────────────────────────────────────────────────────

def get_transforms(
    mode: Literal["train", "val", "test"],
    img_size: int = 224,
) -> T.Compose:
    """Return transforms for the given mode.

    Train: augmentation + normalize.
    Val/Test: deterministic resize-crop + normalize.
    """
    if mode == "train":
        return T.Compose([
            T.Resize(img_size + 32),           # shortest side → img_size+32
            T.RandomCrop(img_size),
            T.RandomHorizontalFlip(),
            T.RandomVerticalFlip(),
            T.RandomApply([T.RandomRotation(90)], p=0.5),
            T.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.1, hue=0.02),
            T.RandomApply([T.GaussianBlur(kernel_size=3)], p=0.2),
            T.ToTensor(),
            T.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ])
    else:
        return T.Compose([
            T.Resize(img_size + 32),
            T.CenterCrop(img_size),
            T.ToTensor(),
            T.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ])


# ── Dataset ──────────────────────────────────────────────────────────────────

def _extract_source_id(filename: str) -> str:
    """Extract the sourceID from a filename like '_0_1169.jpeg' → '0'."""
    parts = filename.split("_")
    if len(parts) >= 3:
        return parts[1]
    return filename  # fallback: treat each file as its own group


class CellDataset(Dataset):
    """Image-folder style dataset with sourceID tracking for group-aware splits."""

    def __init__(
        self,
        root: str | Path,
        split: str = "TRAIN",
        transform: T.Compose | None = None,
    ):
        self.root = Path(root) / split
        self.transform = transform
        self.samples: list[tuple[Path, int]] = []      # (path, label_idx)
        self.source_ids: list[str] = []                 # parallel to samples

        for cls_name in sorted(CLASSES):
            cls_dir = self.root / cls_name
            if not cls_dir.is_dir():
                continue
            for fname in sorted(os.listdir(cls_dir)):
                if not fname.lower().endswith((".jpeg", ".jpg", ".png")):
                    continue
                self.samples.append((cls_dir / fname, CLASS_TO_IDX[cls_name]))
                self.source_ids.append(_extract_source_id(fname))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        path, label = self.samples[idx]
        img = Image.open(path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, label

    @property
    def labels(self) -> np.ndarray:
        return np.array([s[1] for s in self.samples])

    @property
    def groups(self) -> np.ndarray:
        """Numeric group IDs for GroupShuffleSplit."""
        unique = sorted(set(self.source_ids), key=lambda x: int(x) if x.isdigit() else x)
        mapping = {sid: i for i, sid in enumerate(unique)}
        return np.array([mapping[sid] for sid in self.source_ids])


# ── Splits ───────────────────────────────────────────────────────────────────

def get_official_split_loaders(
    data_root: str | Path = DEFAULT_DATA_ROOT,
    img_size: int = 224,
    batch_size: int = 32,
    num_workers: int = 4,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    """Return (train_loader, val_loader, test_loader) using the official split.

    Val is carved from TRAIN (20%, random, NOT group-aware).
    """
    full_train = CellDataset(data_root, "TRAIN", transform=None)
    test_ds = CellDataset(data_root, "TEST", transform=get_transforms("test", img_size))

    # 80/20 random split for train/val
    n = len(full_train)
    indices = np.random.permutation(n)
    split_point = int(0.8 * n)
    train_idx, val_idx = indices[:split_point], indices[split_point:]

    train_ds = _SubsetWithTransform(full_train, train_idx, get_transforms("train", img_size))
    val_ds = _SubsetWithTransform(full_train, val_idx, get_transforms("val", img_size))

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                            num_workers=num_workers, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False,
                             num_workers=num_workers, pin_memory=True)
    return train_loader, val_loader, test_loader


def get_grouped_split_loaders(
    data_root: str | Path = DEFAULT_DATA_ROOT,
    img_size: int = 224,
    batch_size: int = 32,
    num_workers: int = 4,
    val_size: float = 0.15,
    test_size: float = 0.15,
    random_state: int = 42,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    """Return (train, val, test) loaders with group-aware splits.

    All augmentations of the same source cell stay in one split.
    Uses all data from TRAIN + TEST combined, then re-splits by sourceID group.
    """
    # Combine TRAIN and TEST into one pool
    all_train = CellDataset(data_root, "TRAIN", transform=None)
    all_test = CellDataset(data_root, "TEST", transform=None)

    all_samples = all_train.samples + all_test.samples
    all_source_ids = all_train.source_ids + all_test.source_ids
    all_labels = np.array([s[1] for s in all_samples])

    # Numeric group IDs
    unique_groups = sorted(set(all_source_ids), key=lambda x: int(x) if x.isdigit() else x)
    gmap = {sid: i for i, sid in enumerate(unique_groups)}
    groups = np.array([gmap[sid] for sid in all_source_ids])

    # First split: train+val vs test
    gss1 = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
    trainval_idx, test_idx = next(gss1.split(all_samples, all_labels, groups))

    # Second split: train vs val (from trainval)
    rel_val = val_size / (1.0 - test_size)
    trainval_groups = groups[trainval_idx]
    trainval_labels = all_labels[trainval_idx]
    gss2 = GroupShuffleSplit(n_splits=1, test_size=rel_val, random_state=random_state)
    train_sub, val_sub = next(gss2.split(trainval_idx, trainval_labels, trainval_groups))
    train_idx = trainval_idx[train_sub]
    val_idx = trainval_idx[val_sub]

    # Build datasets with proper transforms
    combined = _CombinedDataset(all_samples)
    train_ds = _SubsetWithTransform(combined, train_idx, get_transforms("train", img_size))
    val_ds = _SubsetWithTransform(combined, val_idx, get_transforms("val", img_size))
    test_ds = _SubsetWithTransform(combined, test_idx, get_transforms("test", img_size))

    print(f"Grouped split — train: {len(train_idx)}, val: {len(val_idx)}, test: {len(test_idx)}")
    print(f"  Groups — train: {len(set(groups[train_idx]))}, "
          f"val: {len(set(groups[val_idx]))}, test: {len(set(groups[test_idx]))}")

    # Verify no group leakage
    train_groups = set(groups[train_idx])
    val_groups = set(groups[val_idx])
    test_groups = set(groups[test_idx])
    assert train_groups.isdisjoint(val_groups), "Group leakage: train ∩ val"
    assert train_groups.isdisjoint(test_groups), "Group leakage: train ∩ test"
    assert val_groups.isdisjoint(test_groups), "Group leakage: val ∩ test"

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                            num_workers=num_workers, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False,
                             num_workers=num_workers, pin_memory=True)
    return train_loader, val_loader, test_loader


# ── Helpers ──────────────────────────────────────────────────────────────────

class _CombinedDataset(Dataset):
    """Thin wrapper around a pre-built sample list (no transform)."""

    def __init__(self, samples: list[tuple[Path, int]]):
        self.samples = samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img = Image.open(path).convert("RGB")
        return img, label


class _SubsetWithTransform(Dataset):
    """Subset of a dataset with a different transform applied."""

    def __init__(self, dataset: Dataset, indices: np.ndarray, transform: T.Compose):
        self.dataset = dataset
        self.indices = indices
        self.transform = transform

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        img, label = self.dataset[self.indices[idx]]
        if self.transform:
            img = self.transform(img)
        return img, label


# ── Quick sanity check ───────────────────────────────────────────────────────

if __name__ == "__main__":
    print("Testing official split...")
    train_l, val_l, test_l = get_official_split_loaders(batch_size=4, num_workers=0)
    print(f"  Train batches: {len(train_l)}, Val batches: {len(val_l)}, Test batches: {len(test_l)}")
    x, y = next(iter(train_l))
    print(f"  Batch shape: {x.shape}, Labels: {y}")

    print("\nTesting grouped split...")
    train_l, val_l, test_l = get_grouped_split_loaders(batch_size=4, num_workers=0)
    print(f"  Train batches: {len(train_l)}, Val batches: {len(val_l)}, Test batches: {len(test_l)}")
    x, y = next(iter(train_l))
    print(f"  Batch shape: {x.shape}, Labels: {y}")
