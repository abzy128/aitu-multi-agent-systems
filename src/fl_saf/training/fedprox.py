from __future__ import annotations

import copy

import torch

from fl_saf.evaluation.comms import model_bytes
from fl_saf.training.fedavg import weighted_average
from fl_saf.training.utils import fit_model


def train_fedprox(clients, cfg, dev, model_factory):
    global_model = model_factory().to(dev)
    round_history = []
    comm_per_round = model_bytes(global_model) * len(clients) * 2
    for round_idx in range(1, cfg.federated.n_rounds + 1):
        client_states = []
        weights = []
        round_losses = []
        global_params = {
            k: v.detach().clone().to(dev) for k, v in global_model.named_parameters()
        }
        for client in clients:
            local_model = model_factory().to(dev)
            local_model.load_state_dict(copy.deepcopy(global_model.state_dict()))
            local_model, history = fit_model(
                local_model,
                client.loader("train", cfg.data.batch_size, shuffle=True),
                client.loader("val", cfg.data.batch_size),
                cfg,
                dev,
                global_state=global_params,
                fedprox_mu=cfg.federated.fedprox_mu,
                epochs=cfg.federated.local_epochs,
                patience=cfg.federated.local_epochs + 1,
            )
            client_states.append({k: v.detach().cpu() for k, v in local_model.state_dict().items()})
            weights.append(client.train_size)
            round_losses.append(history[-1]["val_loss"])
        global_model.load_state_dict(weighted_average(client_states, weights))
        round_history.append(
            {
                "round": round_idx,
                "val_loss": sum(round_losses) / len(round_losses),
                "round_bytes": comm_per_round,
                "cumulative_bytes": comm_per_round * round_idx,
            }
        )
    return global_model, round_history, comm_per_round
