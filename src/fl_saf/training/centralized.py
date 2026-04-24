from __future__ import annotations

import torch
from torch.utils.data import DataLoader

from fl_saf.config import ExperimentConfig
from fl_saf.data.loader import ClientData
from fl_saf.evaluation.metrics import evaluate_client
from fl_saf.training.utils import ConcatDataset, build_model, fit_model, metric_rows_for_model


def train_centralized(clients: list[ClientData], cfg: ExperimentConfig, dev: torch.device):
    model = build_model(clients[0].input_size, cfg).to(dev)
    train_loader = DataLoader(
        ConcatDataset([c.train for c in clients]),
        batch_size=cfg.data.batch_size,
        shuffle=True,
    )
    val_loader = DataLoader(
        ConcatDataset([c.val for c in clients]),
        batch_size=cfg.data.batch_size,
        shuffle=False,
    )
    model, history = fit_model(model, train_loader, val_loader, cfg, dev)
    return model, history, metric_rows_for_model(model, clients, "centralized", cfg, dev)


def train_local_only(clients: list[ClientData], cfg: ExperimentConfig, dev: torch.device):
    models = {}
    histories = {}
    rows = []
    for client in clients:
        model = build_model(client.input_size, cfg).to(dev)
        train_loader = client.loader("train", cfg.data.batch_size, shuffle=True)
        val_loader = client.loader("val", cfg.data.batch_size)
        model, history = fit_model(model, train_loader, val_loader, cfg, dev)
        models[client.client_id] = model
        histories[client.client_id] = history
        metrics, _, _ = evaluate_client(model, client, "test", cfg.data.batch_size, dev)
        rows.append({"algorithm": "local_only", "client": client.client_id, **metrics})
    return models, histories, rows
