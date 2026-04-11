"""Evaluation utilities: metrics, threshold selection, ROC, confusion matrix."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from torch.utils.data import DataLoader

from src.config import AE_THRESHOLD_PERCENTILE, CLASS_NAMES, DEVICE


# ── Reconstruction error ───────────────────────────────────────────────────

def compute_reconstruction_errors(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device = DEVICE,
) -> np.ndarray:
    model.eval()
    errors = []
    with torch.no_grad():
        for batch in loader:
            x = batch[0].to(device) if isinstance(batch, (list, tuple)) else batch.to(device)
            x_hat, _ = model(x)
            mse = ((x - x_hat) ** 2).mean(dim=1).cpu().numpy()
            errors.append(mse)
    return np.concatenate(errors)


def select_threshold(
    normal_errors: np.ndarray,
    percentile: float = AE_THRESHOLD_PERCENTILE,
) -> float:
    return float(np.percentile(normal_errors, percentile))


def ae_predict(errors: np.ndarray, threshold: float) -> np.ndarray:
    return (errors > threshold).astype(int)


# ── Classifier inference ───────────────────────────────────────────────────

def classifier_predict(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device = DEVICE,
) -> tuple[np.ndarray, np.ndarray]:
    """Returns (predicted class indices, softmax probabilities)."""
    model.eval()
    preds, probs = [], []
    with torch.no_grad():
        for x, _ in loader:
            x = x.to(device)
            logits = model(x)
            p = torch.softmax(logits, dim=1).cpu().numpy()
            probs.append(p)
            preds.append(logits.argmax(1).cpu().numpy())
    return np.concatenate(preds), np.concatenate(probs)


# ── Metric helpers ─────────────────────────────────────────────────────────

def binary_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
    }


def multiclass_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: np.ndarray,
    class_names: list[str] = CLASS_NAMES,
) -> dict:
    report = classification_report(
        y_true, y_pred, target_names=class_names, output_dict=True, zero_division=0
    )
    try:
        auc = roc_auc_score(y_true, y_prob, multi_class="ovr", average="macro")
    except Exception:
        auc = float("nan")
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "macro_f1": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "roc_auc": auc,
        "per_class": report,
    }


def get_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> np.ndarray:
    return confusion_matrix(y_true, y_pred)


def get_roc_data(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    class_names: list[str] = CLASS_NAMES,
) -> dict:
    """One-vs-rest ROC curves for each class."""
    roc_data = {}
    n_classes = len(class_names)
    y_bin = np.eye(n_classes)[y_true]
    for i, name in enumerate(class_names):
        try:
            fpr, tpr, _ = roc_curve(y_bin[:, i], y_prob[:, i])
            auc = roc_auc_score(y_bin[:, i], y_prob[:, i])
        except Exception:
            fpr, tpr, auc = np.array([0, 1]), np.array([0, 1]), float("nan")
        roc_data[name] = {"fpr": fpr.tolist(), "tpr": tpr.tolist(), "auc": auc}
    return roc_data
