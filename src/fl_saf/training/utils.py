from __future__ import annotations

import copy
import random

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from fl_saf.config import ExperimentConfig
from fl_saf.evaluation.metrics import evaluate_client
from fl_saf.models import LSTMRegressor


def set_seed(seed: int, deterministic: bool = True) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.use_deterministic_algorithms(True, warn_only=True)


def device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def build_model(input_size: int, cfg: ExperimentConfig) -> LSTMRegressor:
    return LSTMRegressor(
        input_size=input_size,
        hidden_size=cfg.model.hidden_size,
        num_layers=cfg.model.num_layers,
        dropout=cfg.model.dropout,
        horizon=cfg.data.horizon,
        use_batchnorm=cfg.model.use_batchnorm or cfg.algo == "fedbn",
    )


def train_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    loss_fn: nn.Module,
    dev: torch.device,
    grad_clip: float,
    global_state: dict[str, torch.Tensor] | None = None,
    fedprox_mu: float = 0.0,
) -> float:
    model.train()
    total = 0.0
    seen = 0
    for x, y in loader:
        x, y = x.to(dev), y.to(dev)
        optimizer.zero_grad(set_to_none=True)
        loss = loss_fn(model(x), y)
        if global_state and fedprox_mu > 0:
            prox = torch.zeros((), device=dev)
            for name, param in model.named_parameters():
                prox = prox + torch.sum((param - global_state[name].to(dev)) ** 2)
            loss = loss + 0.5 * fedprox_mu * prox
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()
        total += float(loss.detach().cpu()) * len(x)
        seen += len(x)
    return total / max(1, seen)


@torch.no_grad()
def eval_loss(model: nn.Module, loader: DataLoader, loss_fn: nn.Module, dev: torch.device) -> float:
    model.eval()
    total = 0.0
    seen = 0
    for x, y in loader:
        x, y = x.to(dev), y.to(dev)
        loss = loss_fn(model(x), y)
        total += float(loss.cpu()) * len(x)
        seen += len(x)
    return total / max(1, seen)


def fit_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    cfg: ExperimentConfig,
    dev: torch.device,
    global_state: dict[str, torch.Tensor] | None = None,
    fedprox_mu: float = 0.0,
    epochs: int | None = None,
    patience: int | None = None,
) -> tuple[nn.Module, list[dict[str, float]]]:
    loss_fn = nn.MSELoss()
    optimizer = torch.optim.Adam(
        model.parameters(), lr=cfg.train.lr, weight_decay=cfg.train.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(1, epochs or cfg.train.epochs)
    )
    best_state = copy.deepcopy(model.state_dict())
    best_val = float("inf")
    wait = 0
    history: list[dict[str, float]] = []
    max_epochs = epochs or cfg.train.epochs
    max_patience = patience if patience is not None else cfg.train.patience
    for epoch in range(1, max_epochs + 1):
        train_loss = train_epoch(
            model, train_loader, optimizer, loss_fn, dev, cfg.train.grad_clip, global_state, fedprox_mu
        )
        val_loss = eval_loss(model, val_loader, loss_fn, dev)
        scheduler.step()
        history.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss})
        if val_loss < best_val:
            best_val = val_loss
            best_state = copy.deepcopy(model.state_dict())
            wait = 0
        else:
            wait += 1
            if wait >= max_patience:
                break
    model.load_state_dict(best_state)
    return model, history


class ConcatDataset(Dataset):
    def __init__(self, datasets: list[Dataset]) -> None:
        self.datasets = datasets
        self.offsets = np.cumsum([0, *[len(d) for d in datasets]])

    def __len__(self) -> int:
        return int(self.offsets[-1])

    def __getitem__(self, idx: int):
        dataset_idx = int(np.searchsorted(self.offsets, idx, side="right") - 1)
        local_idx = idx - int(self.offsets[dataset_idx])
        return self.datasets[dataset_idx][local_idx]


def metric_rows_for_model(model: nn.Module, clients, algo: str, cfg: ExperimentConfig, dev: torch.device):
    rows = []
    for client in clients:
        metrics, _, _ = evaluate_client(model, client, "test", cfg.data.batch_size, dev)
        rows.append({"algorithm": algo, "client": client.client_id, **metrics})
    return rows
