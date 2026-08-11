"""Training helpers shared by ABCD baselines without changing ADNI defaults."""

from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import DataLoader, WeightedRandomSampler


def class_weights(labels, train_ids, power: float, device):
    train_labels = np.asarray(labels)[np.asarray(train_ids)]
    counts = np.bincount(train_labels).astype(np.float64)
    weights = np.power(counts.sum() / np.maximum(counts, 1.0), power)
    weights /= weights.mean()
    return torch.as_tensor(weights, dtype=torch.float32, device=device)


def balanced_train_loader(loader: DataLoader, power: float, seed: int = 0) -> DataLoader:
    if power <= 0:
        return loader
    dataset_labels = np.asarray(loader.dataset.label_new, dtype=np.int64)
    counts = np.bincount(dataset_labels).astype(np.float64)
    per_class = np.power(1.0 / np.maximum(counts, 1.0), power)
    sample_weights = torch.as_tensor(per_class[dataset_labels], dtype=torch.double)
    generator = torch.Generator().manual_seed(int(seed))
    sampler = WeightedRandomSampler(
        sample_weights, num_samples=len(dataset_labels), replacement=True, generator=generator
    )
    return DataLoader(
        loader.dataset,
        batch_size=loader.batch_size,
        sampler=sampler,
        collate_fn=loader.collate_fn,
        num_workers=loader.num_workers,
        pin_memory=loader.pin_memory,
        drop_last=loader.drop_last,
    )
