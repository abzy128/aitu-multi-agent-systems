from __future__ import annotations

import torch

from fl_saf.evaluation.comms import is_batchnorm_key, model_bytes
from fl_saf.training.fedavg import weighted_average
from fl_saf.training.utils import fit_model


def _non_bn_state(state: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    return {k: v for k, v in state.items() if not is_batchnorm_key(k)}


def train_fedbn(clients, cfg, dev, model_factory):
    shared_init = model_factory().to(dev).state_dict()
    client_models = {}
    for c in clients:
        m = model_factory().to(dev)
        m.load_state_dict({k: v.clone() for k, v in shared_init.items()})
        client_models[c.client_id] = m

    comm_per_round = model_bytes(next(iter(client_models.values())), exclude_bn=True) * len(clients) * 2
    round_history = []

    for round_idx in range(1, cfg.federated.n_rounds + 1):
        client_non_bn_states = []
        weights = []
        round_losses = []
        for client in clients:
            local_model = client_models[client.client_id]
            local_model, history = fit_model(
                local_model,
                client.loader("train", cfg.data.batch_size, shuffle=True),
                client.loader("val", cfg.data.batch_size),
                cfg,
                dev,
                epochs=cfg.federated.local_epochs,
                patience=cfg.federated.local_epochs + 1,
            )
            full_state = {k: v.detach().cpu() for k, v in local_model.state_dict().items()}
            client_non_bn_states.append(_non_bn_state(full_state))
            weights.append(client.train_size)
            round_losses.append(history[-1]["val_loss"])

        aggregated_non_bn = weighted_average(client_non_bn_states, weights)
        for client in clients:
            client_models[client.client_id].load_state_dict(aggregated_non_bn, strict=False)

        round_history.append(
            {
                "round": round_idx,
                "val_loss": sum(round_losses) / len(round_losses),
                "round_bytes": comm_per_round,
                "cumulative_bytes": comm_per_round * round_idx,
            }
        )
    return client_models, round_history, comm_per_round
