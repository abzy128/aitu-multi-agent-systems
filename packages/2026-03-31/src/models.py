"""Model architectures: Autoencoder and Classifier."""

from __future__ import annotations

import torch
import torch.nn as nn


class Autoencoder(nn.Module):
    """Symmetric encoder-decoder autoencoder with configurable hidden dims."""

    def __init__(
        self,
        input_dim: int,
        hidden_dims: list[int],
        latent_dim: int,
    ) -> None:
        super().__init__()

        # Encoder
        enc_layers: list[nn.Module] = []
        in_dim = input_dim
        for h in hidden_dims:
            enc_layers += [nn.Linear(in_dim, h), nn.BatchNorm1d(h), nn.ReLU()]
            in_dim = h
        enc_layers.append(nn.Linear(in_dim, latent_dim))
        self.encoder = nn.Sequential(*enc_layers)

        # Decoder (mirrors encoder)
        dec_layers: list[nn.Module] = []
        in_dim = latent_dim
        for h in reversed(hidden_dims):
            dec_layers += [nn.Linear(in_dim, h), nn.BatchNorm1d(h), nn.ReLU()]
            in_dim = h
        dec_layers.append(nn.Linear(in_dim, input_dim))
        self.decoder = nn.Sequential(*dec_layers)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        z = self.encode(x)
        x_hat = self.decode(z)
        return x_hat, z


class Classifier(nn.Module):
    """Feedforward classifier over 5 classes (normal + 4 attack categories)."""

    def __init__(
        self,
        input_dim: int,
        hidden_dims: list[int],
        num_classes: int,
    ) -> None:
        super().__init__()

        layers: list[nn.Module] = []
        in_dim = input_dim
        for h in hidden_dims:
            layers += [nn.Linear(in_dim, h), nn.BatchNorm1d(h), nn.ReLU(), nn.Dropout(0.3)]
            in_dim = h
        layers.append(nn.Linear(in_dim, num_classes))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)
