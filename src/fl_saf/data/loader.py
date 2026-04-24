from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

import torch
from torch.utils.data import DataLoader, TensorDataset

from fl_saf.config import DataConfig
from fl_saf.data.preprocessing import (
    TARGET,
    PreprocessingReport,
    fit_transform_split,
    load_client_csv,
    make_windows,
    select_features,
)
from fl_saf.data.splits import chronological_split


@dataclass(slots=True)
class ClientData:
    client_id: str
    feature_names: list[str]
    train: TensorDataset
    val: TensorDataset
    test: TensorDataset
    report: PreprocessingReport

    @property
    def input_size(self) -> int:
        return len(self.feature_names)

    @property
    def train_size(self) -> int:
        return len(self.train)

    def loader(self, split: str, batch_size: int, shuffle: bool = False) -> DataLoader:
        dataset = getattr(self, split)
        return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

    def inverse_target(self, y: torch.Tensor) -> torch.Tensor:
        return y * self.report.target_scale + self.report.target_mean

    def report_dict(self) -> dict:
        return asdict(self.report)


def build_client_datasets(cfg: DataConfig) -> list[ClientData]:
    data_dir = Path(cfg.data_dir)
    raw_clients = [
        ("client_1", data_dir / "Furnace1.csv"),
        ("client_2", data_dir / "Furnace2.csv"),
    ]
    clients: list[ClientData] = []
    for client_id, path in raw_clients:
        df = load_client_csv(str(path), client_id)
        selected, report = select_features(df, cfg, client_id)
        split = chronological_split(selected, cfg.train_ratio, cfg.val_ratio)
        train_x, val_x, test_x, _, report = fit_transform_split(
            split.train, split.val, split.test, report.feature_names, report
        )
        target_idx = report.feature_names.index(TARGET)
        windows = [
            make_windows(arr, target_idx, cfg.seq_len, cfg.horizon, cfg.stride)
            for arr in (train_x, val_x, test_x)
        ]
        datasets = [
            TensorDataset(torch.from_numpy(x), torch.from_numpy(y)) for x, y in windows
        ]
        clients.append(
            ClientData(
                client_id=client_id,
                feature_names=report.feature_names,
                train=datasets[0],
                val=datasets[1],
                test=datasets[2],
                report=report,
            )
        )
    input_sizes = {c.input_size for c in clients}
    if len(input_sizes) != 1:
        raise ValueError(
            "clients produced different input sizes; use univariate track or a "
            "personalized model path for heterogeneous Track B features"
        )
    return clients
