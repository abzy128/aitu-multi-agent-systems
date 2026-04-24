from __future__ import annotations

import copy

import torch
from torch import nn

from fl_saf.evaluation.comms import is_batchnorm_key, model_bytes
from fl_saf.training.utils import fit_model


def weighted_average(
    states: list[dict[str, torch.Tensor]],
    weights: list[int],
    exclude_bn: bool = False,
) -> dict[str, torch.Tensor]:
    total = float(sum(weights))
    out: dict[str, torch.Tensor] = {}
    for key in states[0]:
        if exclude_bn and is_batchnorm_key(key):
            out[key] = states[0][key].clone()
            continue
        tensor = states[0][key]
        if not torch.is_floating_point(tensor):
            out[key] = tensor.clone()
            continue
        avg = torch.zeros_like(tensor)
        for state, weight in zip(states, weights):
            avg += state[key] * (weight / total)
        out[key] = avg
    return out


def train_fedavg(clients, cfg, dev, model_factory, exclude_bn: bool = False):
    global_model = model_factory().to(dev)
    rows = []
    round_history = []
    comm_per_round = model_bytes(global_model, exclude_bn=exclude_bn) * len(clients) * 2
    for round_idx in range(1, cfg.federated.n_rounds + 1):
        client_states = []
        weights = []
        round_losses = []
        for client in clients:
            local_model = model_factory().to(dev)
            local_model.load_state_dict(copy.deepcopy(global_model.state_dict()), strict=False)
            local_model, history = fit_model(
                local_model,
                client.loader("train", cfg.data.batch_size, shuffle=True),
                client.loader("val", cfg.data.batch_size),
                cfg,
                dev,
                epochs=cfg.federated.local_epochs,
                patience=cfg.federated.local_epochs + 1,
            )
            client_states.append({k: v.detach().cpu() for k, v in local_model.state_dict().items()})
            weights.append(client.train_size)
            round_losses.append(history[-1]["val_loss"])
        new_state = weighted_average(client_states, weights, exclude_bn=exclude_bn)
        global_model.load_state_dict(new_state, strict=False)
        round_history.append(
            {
                "round": round_idx,
                "val_loss": sum(round_losses) / len(round_losses),
                "round_bytes": comm_per_round,
                "cumulative_bytes": comm_per_round * round_idx,
            }
        )
    return global_model, round_history, comm_per_round
