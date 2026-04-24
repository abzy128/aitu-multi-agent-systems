from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True, slots=True)
class ChronologicalSplit:
    train: pd.DataFrame
    val: pd.DataFrame
    test: pd.DataFrame


def chronological_split(
    df: pd.DataFrame, train_ratio: float, val_ratio: float
) -> ChronologicalSplit:
    if not 0 < train_ratio < 1 or not 0 < val_ratio < 1:
        raise ValueError("split ratios must be between 0 and 1")
    if train_ratio + val_ratio >= 1:
        raise ValueError("train_ratio + val_ratio must be < 1")
    train_end = int(len(df) * train_ratio)
    val_end = int(len(df) * (train_ratio + val_ratio))
    return ChronologicalSplit(
        train=df.iloc[:train_end].reset_index(drop=True),
        val=df.iloc[train_end:val_end].reset_index(drop=True),
        test=df.iloc[val_end:].reset_index(drop=True),
    )
