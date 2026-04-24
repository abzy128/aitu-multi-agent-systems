from __future__ import annotations

import math

import numpy as np
import torch
from torch import nn

from fl_saf.data.loader import ClientData


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    err = y_pred - y_true
    mse = float(np.mean(err**2))
    mae = float(np.mean(np.abs(err)))
    denom = np.where(np.abs(y_true) < 1e-8, np.nan, np.abs(y_true))
    mape = float(np.nanmean(np.abs(err) / denom) * 100.0)
    ss_res = float(np.sum(err**2))
    ss_tot = float(np.sum((y_true - np.mean(y_true)) ** 2))
    r2 = float(1.0 - ss_res / ss_tot) if ss_tot > 0 else float("nan")
    return {"rmse": math.sqrt(mse), "mae": mae, "mape": mape, "r2": r2}


@torch.no_grad()
def evaluate_client(
    model: nn.Module,
    client: ClientData,
    split: str,
    batch_size: int,
    device: torch.device,
) -> tuple[dict[str, float], np.ndarray, np.ndarray]:
    model.eval()
    preds: list[torch.Tensor] = []
    ys: list[torch.Tensor] = []
    for x, y in client.loader(split, batch_size=batch_size):
        x = x.to(device)
        pred = model(x).cpu()
        preds.append(pred)
        ys.append(y)
    pred_t = client.inverse_target(torch.cat(preds)).numpy()
    y_t = client.inverse_target(torch.cat(ys)).numpy()
    return regression_metrics(y_t, pred_t), y_t, pred_t
