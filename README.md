# Federated SAF Forecasting

PyTorch experiments for active-power forecasting across two anonymized submerged
arc furnace clients. The main Track A experiment is univariate active-power
forecasting with centralized, local-only, FedAvg, FedProx, SCAFFOLD, and FedBN
baselines.

## Setup

This project uses `uv`.

```bash
uv sync
```

The package exposes the `fl-saf` console script.

## Run Experiments

Smoke run:

```bash
uv run fl-saf --config configs/smoke.yaml --algo centralized --run-id smoke_centralized
```

Track A examples:

```bash
uv run fl-saf --config configs/track_a_run.yaml --algo fedavg --seed 0 --run-id track_a_seed0_fedavg
uv run fl-saf --config configs/track_a_run.yaml --algo fedprox --seed 0 --run-id track_a_seed0_fedprox
uv run fl-saf --config configs/track_a_run.yaml --algo scaffold --seed 0 --run-id track_a_seed0_scaffold
uv run fl-saf --config configs/track_a_run.yaml --algo fedbn --seed 0 --run-id track_a_seed0_fedbn
```

50-round FL ablation:

```bash
uv run fl-saf --config configs/track_a_r50.yaml --algo fedprox --seed 0 --run-id track_a_r50_seed0_fedprox
```

Each run writes to `output/<run_id>/`:

- `config.yaml`
- `preprocessing.json`
- `history.csv`
- `metrics.csv`
- model checkpoint(s)
- `predictions/*_test_predictions.csv`

## Analysis Outputs

Aggregate run metrics:

```bash
uv run python -m fl_saf.evaluation.collect_results --root output --prefix track_a_seed --out output/track_a_summary.csv --stats-out output/track_a_summary_stats.csv
```

Render LaTeX tables and figures:

```bash
uv run python -m fl_saf.evaluation.latex_table --stats output/track_a_summary_stats.csv --out output/track_a_table.tex
uv run python -m fl_saf.evaluation.plots --root output --prefix track_a_seed --summary output/track_a_summary.csv --out-dir output/figures
```

See [TODO.md](TODO.md) for the current result inventory and experiment
decisions. See [docs/guide.md](docs/guide.md) for the implementation and paper
protocol.

## Tests

```bash
uv run pytest
```

The current tests cover chronological splitting, train-only scaling, sequence
windowing, inverse-transformed metrics, communication byte accounting, FedBN
BatchNorm exclusion, and CLI smoke runs for centralized and FedAvg training on
synthetic CSVs.
