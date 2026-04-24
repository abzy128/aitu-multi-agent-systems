from __future__ import annotations

from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any, Literal

import yaml


Track = Literal["univariate", "multivariate"]


@dataclass(slots=True)
class DataConfig:
    data_dir: str = "dataset"
    track: Track = "univariate"
    seq_len: int = 60
    horizon: int = 1
    stride: int = 1
    train_ratio: float = 0.70
    val_ratio: float = 0.15
    constant_eps: float = 1e-3
    batch_size: int = 64
    num_workers: int = 0


@dataclass(slots=True)
class ModelConfig:
    hidden_size: int = 64
    num_layers: int = 2
    dropout: float = 0.2
    use_batchnorm: bool = False


@dataclass(slots=True)
class TrainConfig:
    epochs: int = 100
    patience: int = 10
    lr: float = 1e-3
    weight_decay: float = 1e-5
    grad_clip: float = 1.0
    deterministic: bool = True


@dataclass(slots=True)
class FederatedConfig:
    n_rounds: int = 50
    local_epochs: int = 2
    local_lr: float = 1e-3
    fedprox_mu: float = 0.01


@dataclass(slots=True)
class ExperimentConfig:
    seed: int = 0
    run_id: str = "default"
    output_dir: str = "experiments"
    algo: str = "centralized"
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    federated: FederatedConfig = field(default_factory=FederatedConfig)

    @classmethod
    def from_yaml(cls, path: str | Path) -> "ExperimentConfig":
        raw = yaml.safe_load(Path(path).read_text()) or {}
        return cls.from_dict(raw)

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "ExperimentConfig":
        kwargs: dict[str, Any] = {}
        for f in fields(cls):
            value = raw.get(f.name)
            if value is None:
                continue
            if f.name == "data":
                value = DataConfig(**value)
            elif f.name == "model":
                value = ModelConfig(**value)
            elif f.name == "train":
                value = TrainConfig(**value)
            elif f.name == "federated":
                value = FederatedConfig(**value)
            kwargs[f.name] = value
        return cls(**kwargs)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def dump_yaml(self, path: str | Path) -> None:
        Path(path).write_text(yaml.safe_dump(self.to_dict(), sort_keys=False))
