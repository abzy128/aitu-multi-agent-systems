from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import torch

from fl_saf.config import ExperimentConfig
from fl_saf.data import build_client_datasets
from fl_saf.evaluation.metrics import evaluate_client
from fl_saf.training.centralized import train_centralized, train_local_only
from fl_saf.training.federated import train_federated
from fl_saf.training.utils import device, set_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run SAF LSTM experiments")
    parser.add_argument("--algo", choices=["centralized", "local_only", "fedavg", "fedprox", "scaffold", "fedbn"])
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--seed", type=int)
    parser.add_argument("--run-id")
    return parser.parse_args()


def write_rows(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    keys = sorted({k for row in rows for k in row})
    with path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def write_history(path: Path, history) -> None:
    if isinstance(history, dict):
        rows = []
        for client, entries in history.items():
            rows.extend({"client": client, **entry} for entry in entries)
    else:
        rows = history
    write_rows(path, rows)


def write_predictions(run_dir: Path, model, clients, cfg: ExperimentConfig, dev: torch.device) -> None:
    pred_dir = run_dir / "predictions"
    pred_dir.mkdir(exist_ok=True)
    for client in clients:
        eval_model = model[client.client_id] if isinstance(model, dict) else model
        _, y_true, y_pred = evaluate_client(eval_model, client, "test", cfg.data.batch_size, dev)
        rows = []
        for idx, (truth_row, pred_row) in enumerate(zip(y_true, y_pred)):
            row = {"index": idx, "client": client.client_id}
            for horizon_idx, value in enumerate(truth_row, start=1):
                row[f"y_true_t+{horizon_idx}"] = float(value)
            for horizon_idx, value in enumerate(pred_row, start=1):
                row[f"y_pred_t+{horizon_idx}"] = float(value)
            rows.append(row)
        write_rows(pred_dir / f"{client.client_id}_test_predictions.csv", rows)


def main() -> None:
    args = parse_args()
    cfg = ExperimentConfig.from_yaml(args.config)
    if args.algo:
        cfg.algo = args.algo
    if args.seed is not None:
        cfg.seed = args.seed
    if args.run_id:
        cfg.run_id = args.run_id
    if cfg.algo == "fedbn":
        cfg.model.use_batchnorm = True

    set_seed(cfg.seed, cfg.train.deterministic)
    run_dir = Path(cfg.output_dir) / cfg.run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    cfg.dump_yaml(run_dir / "config.yaml")

    clients = build_client_datasets(cfg.data)
    (run_dir / "preprocessing.json").write_text(
        json.dumps([c.report_dict() for c in clients], indent=2)
    )

    dev = device()
    if cfg.algo == "centralized":
        model, history, rows = train_centralized(clients, cfg, dev)
    elif cfg.algo == "local_only":
        model, history, rows = train_local_only(clients, cfg, dev)
    else:
        model, history, rows = train_federated(clients, cfg, dev)

    write_history(run_dir / "history.csv", history)
    write_rows(run_dir / "metrics.csv", rows)
    write_predictions(run_dir, model, clients, cfg, dev)
    if isinstance(model, dict):
        for client_id, local_model in model.items():
            torch.save(local_model.state_dict(), run_dir / f"{client_id}_model.pt")
    else:
        torch.save(model.state_dict(), run_dir / "model.pt")
    print(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()
