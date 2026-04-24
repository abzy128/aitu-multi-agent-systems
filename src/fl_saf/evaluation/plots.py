from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


FL_ALGOS = ["fedavg", "fedprox", "scaffold", "fedbn"]
ALL_ALGOS = ["centralized", "local_only", *FL_ALGOS]
ALGO_LABELS = {
    "centralized": "Centralized",
    "local_only": "Local only",
    "fedavg": "FedAvg",
    "fedprox": "FedProx",
    "scaffold": "SCAFFOLD",
    "fedbn": "FedBN",
}
ALGO_COLORS = {
    "centralized": "#444444",
    "local_only": "#888888",
    "fedavg": "#1f77b4",
    "fedprox": "#2ca02c",
    "scaffold": "#d62728",
    "fedbn": "#9467bd",
}


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as fh:
        return list(csv.DictReader(fh))


def plot_val_loss(root: Path, prefix: str, out: Path) -> None:
    fig, ax = plt.subplots(figsize=(6.5, 4.0))
    for algo in FL_ALGOS:
        curves = []
        for run_dir in sorted(root.glob(f"{prefix}*_{algo}")):
            hist = read_csv_rows(run_dir / "history.csv")
            rounds = np.array([int(float(r["round"])) for r in hist])
            loss = np.array([float(r["val_loss"]) for r in hist])
            order = np.argsort(rounds)
            curves.append((rounds[order], loss[order]))
        if not curves:
            continue
        min_len = min(len(r) for r, _ in curves)
        stacked = np.stack([l[:min_len] for _, l in curves])
        mean = stacked.mean(axis=0)
        std = stacked.std(axis=0)
        x = curves[0][0][:min_len]
        ax.plot(x, mean, label=ALGO_LABELS[algo], color=ALGO_COLORS[algo], linewidth=1.8)
        ax.fill_between(x, mean - std, mean + std, color=ALGO_COLORS[algo], alpha=0.15)
    ax.set_xlabel("Communication round")
    ax.set_ylabel("Validation MSE (standardized)")
    ax.set_title(f"FL validation loss vs. round ({prefix})")
    ax.legend(loc="upper right", frameon=False)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"wrote {out}")


def plot_predictions(root: Path, prefix: str, out: Path, window: int, seed: int) -> None:
    fig, axes = plt.subplots(
        len(ALL_ALGOS), 2,
        figsize=(10.5, 2.0 * len(ALL_ALGOS)),
        sharex="col",
    )
    for row, algo in enumerate(ALL_ALGOS):
        run_dir = root / f"{prefix}{seed}_{algo}"
        if not run_dir.exists():
            for col in (0, 1):
                axes[row, col].set_axis_off()
            continue
        for col, client in enumerate(("client_1", "client_2")):
            ax = axes[row, col]
            pred_path = run_dir / "predictions" / f"{client}_test_predictions.csv"
            rows = read_csv_rows(pred_path)
            y_true = np.array([float(r["y_true_t+1"]) for r in rows])
            y_pred = np.array([float(r["y_pred_t+1"]) for r in rows])
            n = min(window, len(y_true))
            idx = np.arange(n)
            ax.plot(idx, y_true[:n], label="truth", color="black", linewidth=1.0)
            ax.plot(idx, y_pred[:n], label="pred", color=ALGO_COLORS[algo], linewidth=1.0, alpha=0.85)
            if col == 0:
                ax.set_ylabel(ALGO_LABELS[algo], fontsize=9)
            if row == 0:
                ax.set_title(f"Client {client[-1]}")
            ax.grid(True, alpha=0.3)
            ax.tick_params(labelsize=8)
    axes[-1, 0].set_xlabel("Test-window minute")
    axes[-1, 1].set_xlabel("Test-window minute")
    axes[0, 1].legend(loc="upper right", fontsize=8, frameon=False)
    fig.suptitle(f"Test-set predictions, first {window} min (seed {seed}, {prefix})", y=1.0)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"wrote {out}")


def plot_pareto(summary_csv: Path, out: Path) -> None:
    rows = read_csv_rows(summary_csv)
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.2), sharey=False)
    for col, client in enumerate(("client_1", "client_2")):
        ax = axes[col]
        for algo in FL_ALGOS:
            xs = []
            ys = []
            for r in rows:
                if r["algorithm"] != algo or r["client"] != client:
                    continue
                if r.get("comm_mb", "") in ("", None):
                    continue
                xs.append(float(r["comm_mb"]))
                ys.append(float(r["rmse"]))
            if xs:
                ax.scatter(xs, ys, label=ALGO_LABELS[algo], color=ALGO_COLORS[algo], s=40, alpha=0.85)
        ax.set_xlabel("Communication cost (MB)")
        ax.set_ylabel("Test RMSE (MW)")
        ax.set_title(f"Client {client[-1]}")
        ax.grid(True, alpha=0.3)
    axes[0].legend(loc="upper right", frameon=False)
    fig.suptitle(f"Communication vs. RMSE Pareto ({summary_csv.stem})")
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"wrote {out}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render Track A figures")
    parser.add_argument("--kind", choices=["val_loss", "predictions", "pareto", "all"], default="all")
    parser.add_argument("--root", default="output")
    parser.add_argument("--prefix", default="track_a_seed")
    parser.add_argument("--summary", default="output/track_a_summary.csv")
    parser.add_argument("--out-dir", default="output/figures")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--window", type=int, default=720)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = Path(args.root)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    tag = args.prefix.rstrip("_")
    if args.kind in ("val_loss", "all"):
        plot_val_loss(root, args.prefix, out_dir / f"{tag}_val_loss.png")
    if args.kind in ("predictions", "all"):
        plot_predictions(root, args.prefix, out_dir / f"{tag}_predictions_seed{args.seed}.png", args.window, args.seed)
    if args.kind in ("pareto", "all"):
        plot_pareto(Path(args.summary), out_dir / f"{tag}_pareto.png")


if __name__ == "__main__":
    main()
