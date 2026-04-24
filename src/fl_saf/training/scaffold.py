from __future__ import annotations

import copy

import torch
from torch import nn

from fl_saf.evaluation.comms import model_bytes
from fl_saf.training.fedavg import weighted_average
from fl_saf.training.utils import eval_loss


def _float_param_state(model):
    return {
        k: v.detach().clone()
        for k, v in model.state_dict().items()
        if torch.is_floating_point(v)
    }


def train_scaffold(clients, cfg, dev, model_factory):
    global_model = model_factory().to(dev)
    server_c = {k: torch.zeros_like(v).to(dev) for k, v in _float_param_state(global_model).items()}
    client_c = {
        client.client_id: {k: torch.zeros_like(v).to(dev) for k, v in server_c.items()}
        for client in clients
    }
    round_bytes = model_bytes(global_model) * len(clients) * 4
    round_history = []
    loss_fn = nn.MSELoss()

    for round_idx in range(1, cfg.federated.n_rounds + 1):
        global_state = copy.deepcopy(global_model.state_dict())
        global_float = _float_param_state(global_model)
        client_states = []
        c_deltas = []
        weights = []
        val_losses = []
        for client in clients:
            local_model = model_factory().to(dev)
            local_model.load_state_dict(global_state)
            optimizer = torch.optim.SGD(
                local_model.parameters(),
                lr=cfg.federated.scaffold_lr,
                momentum=cfg.federated.scaffold_momentum,
                weight_decay=cfg.train.weight_decay,
            )
            steps = 0
            local_model.train()
            for _ in range(cfg.federated.local_epochs):
                for x, y in client.loader("train", cfg.data.batch_size, shuffle=True):
                    x, y = x.to(dev), y.to(dev)
                    optimizer.zero_grad(set_to_none=True)
                    loss = loss_fn(local_model(x), y)
                    loss.backward()
                    for name, param in local_model.named_parameters():
                        if param.grad is not None:
                            param.grad.add_(server_c[name] - client_c[client.client_id][name])
                    nn.utils.clip_grad_norm_(local_model.parameters(), cfg.train.grad_clip)
                    optimizer.step()
                    steps += 1
            local_float = _float_param_state(local_model)
            new_client_c = {}
            delta_c = {}
            scale = 1.0 / max(1, steps) / cfg.federated.scaffold_lr
            for name in server_c:
                updated = client_c[client.client_id][name] - server_c[name] + (
                    global_float[name].to(dev) - local_float[name].to(dev)
                ) * scale
                new_client_c[name] = updated.detach()
                delta_c[name] = updated.detach() - client_c[client.client_id][name]
            client_c[client.client_id] = new_client_c
            c_deltas.append(delta_c)
            client_states.append({k: v.detach().cpu() for k, v in local_model.state_dict().items()})
            weights.append(client.train_size)
            val_losses.append(
                eval_loss(local_model, client.loader("val", cfg.data.batch_size), loss_fn, dev)
            )

        global_model.load_state_dict(weighted_average(client_states, weights))
        for name in server_c:
            server_c[name] = server_c[name] + sum(d[name] for d in c_deltas) / len(c_deltas)
        round_history.append(
            {
                "round": round_idx,
                "val_loss": sum(val_losses) / len(val_losses),
                "round_bytes": round_bytes,
                "cumulative_bytes": round_bytes * round_idx,
            }
        )
    return global_model, round_history, round_bytes
