from __future__ import annotations

import torch

from fl_saf.evaluation.metrics import evaluate_client
from fl_saf.training.fedavg import train_fedavg
from fl_saf.training.fedbn import train_fedbn
from fl_saf.training.fedprox import train_fedprox
from fl_saf.training.scaffold import train_scaffold
from fl_saf.training.utils import build_model


def train_federated(clients, cfg, dev: torch.device):
    def model_factory():
        return build_model(clients[0].input_size, cfg)

    trainers = {
        "fedavg": train_fedavg,
        "fedprox": train_fedprox,
        "scaffold": train_scaffold,
        "fedbn": train_fedbn,
    }
    if cfg.algo not in trainers:
        raise ValueError(f"unknown federated algorithm: {cfg.algo}")
    model, history, round_bytes = trainers[cfg.algo](clients, cfg, dev, model_factory)
    rows = []
    total_mb = (round_bytes * cfg.federated.n_rounds) / 1_000_000
    for client in clients:
        metrics, _, _ = evaluate_client(model, client, "test", cfg.data.batch_size, dev)
        rows.append(
            {
                "algorithm": cfg.algo,
                "client": client.client_id,
                **metrics,
                "comm_rounds": cfg.federated.n_rounds,
                "comm_mb": total_mb,
            }
        )
    return model, history, rows
