"""PyTorch Dataset and DataLoader factories for NSL-KDD."""

from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from src.config import BATCH_SIZE, NUM_WORKERS


class NetworkTrafficDataset(Dataset):
    """Wraps numpy feature and label arrays as a PyTorch Dataset."""

    def __init__(
        self,
        X: np.ndarray,
        y: np.ndarray | None = None,
    ) -> None:
        self.X = torch.from_numpy(X)
        self.y = torch.from_numpy(y) if y is not None else None

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, idx: int):
        if self.y is not None:
            return self.X[idx], self.y[idx]
        return self.X[idx]


def make_loader(
    X: np.ndarray,
    y: np.ndarray | None = None,
    batch_size: int = BATCH_SIZE,
    shuffle: bool = True,
    num_workers: int = NUM_WORKERS,
) -> DataLoader:
    dataset = NetworkTrafficDataset(X, y)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
    )
