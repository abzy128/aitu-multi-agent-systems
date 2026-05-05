from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from fl_saf.experiments.run import main


def _write_synthetic_client(path: Path, offset: float) -> None:
    rows = 80
    minutes = pd.date_range("2026-01-01", periods=rows, freq="min")
    signal = offset + np.sin(np.arange(rows, dtype=np.float32) / 5.0)
    frame = pd.DataFrame({"DateTime": minutes, "active_power": signal})
    frame.to_csv(path, index=False)


def _write_config(path: Path, data_dir: Path, output_dir: Path, algo: str) -> None:
    path.write_text(
        yaml.safe_dump(
            {
                "seed": 0,
                "run_id": f"smoke_{algo}",
                "output_dir": str(output_dir),
                "algo": algo,
                "data": {
                    "data_dir": str(data_dir),
                    "track": "univariate",
                    "seq_len": 4,
                    "horizon": 1,
                    "stride": 2,
                    "train_ratio": 0.70,
                    "val_ratio": 0.15,
                    "constant_eps": 0.001,
                    "batch_size": 8,
                    "num_workers": 0,
                },
                "model": {
                    "hidden_size": 4,
                    "num_layers": 1,
                    "dropout": 0.0,
                    "use_batchnorm": False,
                },
                "train": {
                    "epochs": 1,
                    "patience": 1,
                    "lr": 0.001,
                    "weight_decay": 0.00001,
                    "grad_clip": 1.0,
                    "deterministic": True,
                },
                "federated": {
                    "n_rounds": 1,
                    "local_epochs": 1,
                    "local_lr": 0.001,
                    "fedprox_mu": 0.01,
                    "scaffold_lr": 0.01,
                    "scaffold_momentum": 0.9,
                },
            },
            sort_keys=False,
        )
    )


@pytest.mark.parametrize("algo", ["centralized", "fedavg"])
def test_cli_smoke_writes_expected_run_artifacts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, algo: str
) -> None:
    data_dir = tmp_path / "dataset"
    output_dir = tmp_path / "output"
    data_dir.mkdir()
    _write_synthetic_client(data_dir / "Furnace1.csv", offset=10.0)
    _write_synthetic_client(data_dir / "Furnace2.csv", offset=20.0)
    config_path = tmp_path / f"{algo}.yaml"
    _write_config(config_path, data_dir, output_dir, algo)

    monkeypatch.setattr(
        sys,
        "argv",
        ["fl-saf", "--config", str(config_path), "--algo", algo],
    )

    main()

    run_dir = output_dir / f"smoke_{algo}"
    assert (run_dir / "config.yaml").is_file()
    assert (run_dir / "preprocessing.json").is_file()
    assert (run_dir / "history.csv").is_file()
    assert (run_dir / "metrics.csv").is_file()
    assert (run_dir / "model.pt").is_file()
    assert (run_dir / "predictions" / "client_1_test_predictions.csv").is_file()
    assert (run_dir / "predictions" / "client_2_test_predictions.csv").is_file()

    reports = json.loads((run_dir / "preprocessing.json").read_text())
    assert [report["client_id"] for report in reports] == ["client_1", "client_2"]
    assert all(report["feature_names"] == ["active_power"] for report in reports)

    with (run_dir / "metrics.csv").open(newline="") as fh:
        metrics = list(csv.DictReader(fh))
    assert len(metrics) == 2
    assert {row["client"] for row in metrics} == {"client_1", "client_2"}
    assert all(row["algorithm"] == algo for row in metrics)
    assert all(float(row["rmse"]) >= 0.0 for row in metrics)
