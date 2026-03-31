"""Plotting helpers — all figures saved to artifacts/."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.manifold import TSNE

from src.config import ARTIFACTS_DIR, CLASS_NAMES

plt.rcParams.update({"figure.dpi": 120, "figure.facecolor": "white"})


def _save(fig: plt.Figure, name: str, artifacts_dir: Path = ARTIFACTS_DIR) -> Path:
    path = artifacts_dir / name
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {path}")
    return path


def plot_loss_curves(
    history: dict,
    title: str = "Loss Curves",
    filename: str = "loss_curves.png",
) -> Path:
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(history["train_loss"], label="Train Loss")
    ax.plot(history["val_loss"], label="Val Loss")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)
    return _save(fig, filename)


def plot_accuracy_curves(history: dict, filename: str = "accuracy_curves.png") -> Path:
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(history["train_acc"], label="Train Accuracy")
    ax.plot(history["val_acc"], label="Val Accuracy")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Accuracy")
    ax.set_title("Accuracy Curves")
    ax.legend()
    ax.grid(True, alpha=0.3)
    return _save(fig, filename)


def plot_confusion_matrix(
    cm: np.ndarray,
    class_names: list[str] = CLASS_NAMES,
    filename: str = "confusion_matrix.png",
) -> Path:
    fig, ax = plt.subplots(figsize=(7, 6))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=class_names, yticklabels=class_names, ax=ax,
    )
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("Confusion Matrix")
    return _save(fig, filename)


def plot_roc_curve(
    roc_data: dict,
    filename: str = "roc_curves.png",
) -> Path:
    fig, ax = plt.subplots(figsize=(8, 6))
    for cls_name, data in roc_data.items():
        ax.plot(
            data["fpr"], data["tpr"],
            label=f"{cls_name} (AUC={data['auc']:.3f})"
        )
    ax.plot([0, 1], [0, 1], "k--", alpha=0.5)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curves (One-vs-Rest)")
    ax.legend(loc="lower right")
    ax.grid(True, alpha=0.3)
    return _save(fig, filename)


def plot_reconstruction_error_dist(
    normal_errors: np.ndarray,
    attack_errors: np.ndarray,
    threshold: float,
    filename: str = "recon_error_dist.png",
) -> Path:
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.hist(normal_errors, bins=100, alpha=0.6, label="Normal", color="steelblue", density=True)
    ax.hist(attack_errors, bins=100, alpha=0.6, label="Attack", color="tomato", density=True)
    ax.axvline(threshold, color="black", linestyle="--", label=f"Threshold={threshold:.4f}")
    ax.set_xlabel("Reconstruction Error (MSE)")
    ax.set_ylabel("Density")
    ax.set_title("Reconstruction Error Distribution")
    ax.legend()
    ax.grid(True, alpha=0.3)
    return _save(fig, filename)


def plot_tsne(
    embeddings: np.ndarray,
    labels: np.ndarray,
    class_names: list[str] = CLASS_NAMES,
    filename: str = "tsne_latent.png",
    random_state: int = 42,
) -> Path:
    print("Running t-SNE (this may take a minute)...")
    # Sub-sample if large
    n = min(len(embeddings), 5000)
    idx = np.random.RandomState(random_state).choice(len(embeddings), n, replace=False)
    emb_sub = embeddings[idx]
    lab_sub = labels[idx]

    tsne = TSNE(n_components=2, random_state=random_state, perplexity=30)
    coords = tsne.fit_transform(emb_sub)

    fig, ax = plt.subplots(figsize=(9, 7))
    palette = plt.cm.tab10.colors
    for i, name in enumerate(class_names):
        mask = lab_sub == i
        ax.scatter(coords[mask, 0], coords[mask, 1],
                   s=6, alpha=0.5, label=name, color=palette[i % len(palette)])
    ax.set_title("t-SNE of Autoencoder Latent Space")
    ax.legend(markerscale=3)
    ax.axis("off")
    return _save(fig, filename)


def plot_class_distribution(
    counts: dict,
    title: str = "Class Distribution",
    filename: str = "class_distribution.png",
) -> Path:
    fig, ax = plt.subplots(figsize=(10, 5))
    labels = list(counts.keys())
    values = list(counts.values())
    bars = ax.bar(labels, values, color=plt.cm.tab10.colors[:len(labels)])
    ax.set_xlabel("Class")
    ax.set_ylabel("Count")
    ax.set_title(title)
    ax.tick_params(axis="x", rotation=45)
    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                f"{val:,}", ha="center", va="bottom", fontsize=8)
    ax.grid(True, alpha=0.3, axis="y")
    return _save(fig, filename)


def plot_correlation_heatmap(
    corr_matrix: np.ndarray,
    feature_names: list[str],
    filename: str = "correlation_heatmap.png",
) -> Path:
    fig, ax = plt.subplots(figsize=(14, 12))
    sns.heatmap(corr_matrix, xticklabels=feature_names, yticklabels=feature_names,
                cmap="coolwarm", center=0, ax=ax, linewidths=0.1)
    ax.set_title("Feature Correlation Heatmap")
    ax.tick_params(axis="x", rotation=90, labelsize=7)
    ax.tick_params(axis="y", rotation=0, labelsize=7)
    return _save(fig, filename)
