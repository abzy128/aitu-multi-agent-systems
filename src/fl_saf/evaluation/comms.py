from __future__ import annotations

import torch
from torch import nn


def tensor_bytes(t: torch.Tensor) -> int:
    return t.numel() * t.element_size()


def state_dict_bytes(state: dict[str, torch.Tensor], exclude_bn: bool = False) -> int:
    return sum(
        tensor_bytes(v)
        for k, v in state.items()
        if not (exclude_bn and is_batchnorm_key(k)) and torch.is_tensor(v)
    )


def model_bytes(model: nn.Module, exclude_bn: bool = False) -> int:
    return state_dict_bytes(model.state_dict(), exclude_bn=exclude_bn)


def is_batchnorm_key(name: str) -> bool:
    return name.startswith("bn.")
