"""Training utilities: training loop, early stopping, checkpointing."""

from __future__ import annotations

from pathlib import Path
from typing import Callable

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm


# ── Early Stopping ─────────────────────────────────────────────────────────

class EarlyStopping:
    def __init__(self, patience: int = 7, min_delta: float = 1e-4) -> None:
        self.patience = patience
        self.min_delta = min_delta
        self.best_loss = float("inf")
        self.counter = 0
        self.should_stop = False

    def step(self, val_loss: float) -> bool:
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True
        return self.should_stop


# ── Checkpoint helpers ─────────────────────────────────────────────────────

def save_checkpoint(model: nn.Module, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), path)


def load_checkpoint(model: nn.Module, path: Path, device: torch.device) -> nn.Module:
    model.load_state_dict(torch.load(path, map_location=device))
    return model


# ── Autoencoder training ───────────────────────────────────────────────────

def train_autoencoder_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> float:
    model.train()
    criterion = nn.MSELoss()
    total_loss = 0.0
    for batch in loader:
        x = batch[0].to(device) if isinstance(batch, (list, tuple)) else batch.to(device)
        optimizer.zero_grad()
        x_hat, _ = model(x)
        loss = criterion(x_hat, x)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * x.size(0)
    return total_loss / len(loader.dataset)


def validate_autoencoder(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> float:
    model.eval()
    criterion = nn.MSELoss()
    total_loss = 0.0
    with torch.no_grad():
        for batch in loader:
            x = batch[0].to(device) if isinstance(batch, (list, tuple)) else batch.to(device)
            x_hat, _ = model(x)
            loss = criterion(x_hat, x)
            total_loss += loss.item() * x.size(0)
    return total_loss / len(loader.dataset)


def train_autoencoder(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epochs: int,
    patience: int,
    checkpoint_path: Path,
) -> dict:
    es = EarlyStopping(patience=patience)
    history = {"train_loss": [], "val_loss": []}
    best_val = float("inf")

    for epoch in range(1, epochs + 1):
        tr_loss = train_autoencoder_epoch(model, train_loader, optimizer, device)
        val_loss = validate_autoencoder(model, val_loader, device)
        history["train_loss"].append(tr_loss)
        history["val_loss"].append(val_loss)

        if val_loss < best_val:
            best_val = val_loss
            save_checkpoint(model, checkpoint_path)

        print(f"Epoch {epoch:3d}/{epochs}  train_loss={tr_loss:.6f}  val_loss={val_loss:.6f}")

        if es.step(val_loss):
            print(f"Early stopping at epoch {epoch}")
            break

    return history


# ── Classifier training ────────────────────────────────────────────────────

def train_classifier_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> tuple[float, float]:
    model.train()
    criterion = nn.CrossEntropyLoss()
    total_loss = 0.0
    correct = 0
    total = 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        logits = model(x)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * x.size(0)
        correct += (logits.argmax(1) == y).sum().item()
        total += x.size(0)
    return total_loss / total, correct / total


def validate_classifier(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> tuple[float, float]:
    model.eval()
    criterion = nn.CrossEntropyLoss()
    total_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            loss = criterion(logits, y)
            total_loss += loss.item() * x.size(0)
            correct += (logits.argmax(1) == y).sum().item()
            total += x.size(0)
    return total_loss / total, correct / total


def train_classifier(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epochs: int,
    patience: int,
    checkpoint_path: Path,
) -> dict:
    es = EarlyStopping(patience=patience)
    history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}
    best_val = float("inf")

    for epoch in range(1, epochs + 1):
        tr_loss, tr_acc = train_classifier_epoch(model, train_loader, optimizer, device)
        val_loss, val_acc = validate_classifier(model, val_loader, device)
        history["train_loss"].append(tr_loss)
        history["val_loss"].append(val_loss)
        history["train_acc"].append(tr_acc)
        history["val_acc"].append(val_acc)

        if val_loss < best_val:
            best_val = val_loss
            save_checkpoint(model, checkpoint_path)

        print(
            f"Epoch {epoch:3d}/{epochs}  "
            f"train_loss={tr_loss:.4f}  val_loss={val_loss:.4f}  "
            f"train_acc={tr_acc:.4f}  val_acc={val_acc:.4f}"
        )

        if es.step(val_loss):
            print(f"Early stopping at epoch {epoch}")
            break

    return history
