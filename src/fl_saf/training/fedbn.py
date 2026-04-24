from __future__ import annotations

from fl_saf.training.fedavg import train_fedavg


def train_fedbn(clients, cfg, dev, model_factory):
    return train_fedavg(clients, cfg, dev, model_factory, exclude_bn=True)
