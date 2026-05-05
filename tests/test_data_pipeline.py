from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from fl_saf.data.preprocessing import (
    TARGET,
    PreprocessingReport,
    fit_transform_split,
    make_windows,
)
from fl_saf.data.splits import chronological_split


def test_chronological_split_preserves_order_and_sizes() -> None:
    frame = pd.DataFrame(
        {
            "DateTime": pd.date_range("2026-01-01", periods=10, freq="min"),
            TARGET: np.arange(10, dtype=float),
        }
    )

    split = chronological_split(frame, train_ratio=0.6, val_ratio=0.2)

    assert split.train[TARGET].tolist() == [0, 1, 2, 3, 4, 5]
    assert split.val[TARGET].tolist() == [6, 7]
    assert split.test[TARGET].tolist() == [8, 9]


@pytest.mark.parametrize(
    ("train_ratio", "val_ratio"),
    [(0.0, 0.2), (0.7, 0.0), (0.8, 0.2), (1.2, 0.1)],
)
def test_chronological_split_rejects_invalid_ratios(
    train_ratio: float, val_ratio: float
) -> None:
    frame = pd.DataFrame({TARGET: np.arange(5, dtype=float)})

    with pytest.raises(ValueError):
        chronological_split(frame, train_ratio=train_ratio, val_ratio=val_ratio)


def test_fit_transform_split_uses_train_statistics_only() -> None:
    train = pd.DataFrame({TARGET: [10.0, 12.0, 14.0], "feature": [1.0, 2.0, 3.0]})
    val = pd.DataFrame({TARGET: [16.0], "feature": [4.0]})
    test = pd.DataFrame({TARGET: [18.0], "feature": [5.0]})
    report = PreprocessingReport("client_1", "univariate", [TARGET, "feature"])

    train_x, val_x, test_x, _, report = fit_transform_split(
        train, val, test, [TARGET, "feature"], report
    )

    assert np.allclose(train_x.mean(axis=0), [0.0, 0.0], atol=1e-6)
    assert np.allclose(report.scaler_mean, [12.0, 2.0])
    assert np.allclose(report.scaler_scale, np.std([[10.0, 1.0], [12.0, 2.0], [14.0, 3.0]], axis=0))
    assert np.allclose(val_x[0], [(16.0 - 12.0) / report.target_scale, (4.0 - 2.0) / report.scaler_scale[1]])
    assert np.allclose(test_x[0], [(18.0 - 12.0) / report.target_scale, (5.0 - 2.0) / report.scaler_scale[1]])
    assert report.target_mean == 12.0
    assert report.target_scale == report.scaler_scale[0]


def test_make_windows_respects_sequence_horizon_and_stride() -> None:
    values = np.arange(8, dtype=np.float32).reshape(-1, 1)

    x, y = make_windows(values, target_index=0, seq_len=3, horizon=2, stride=2)

    assert x.shape == (2, 3, 1)
    assert y.shape == (2, 2)
    assert x[:, :, 0].tolist() == [[0.0, 1.0, 2.0], [2.0, 3.0, 4.0]]
    assert y.tolist() == [[3.0, 4.0], [5.0, 6.0]]


def test_make_windows_rejects_too_short_series() -> None:
    values = np.arange(3, dtype=np.float32).reshape(-1, 1)

    with pytest.raises(ValueError, match="not enough rows"):
        make_windows(values, target_index=0, seq_len=3, horizon=1, stride=1)
