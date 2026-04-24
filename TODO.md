# TODO

## Implemented

- Track A univariate active-power forecasting pipeline.
- Dataclass/YAML experiment configuration.
- Chronological train/validation/test splits with train-only per-client scaling.
- LSTM centralized and local-only baselines.
- Federated algorithms: FedAvg, FedProx, SCAFFOLD, and FedBN.
- Communication accounting for federated runs.
- Per-client metrics and prediction CSV export.
- Aggregate metric collection script.

## Current Outputs

Track A seed-0 runs were executed and saved under `output/`:

- `output/track_a_seed0_centralized/`
- `output/track_a_seed0_local_only/`
- `output/track_a_seed0_fedavg/`
- `output/track_a_seed0_fedprox/`
- `output/track_a_seed0_scaffold/`
- `output/track_a_seed0_fedbn/`

Each run contains:

- `config.yaml`
- `preprocessing.json`
- `history.csv`
- `metrics.csv`
- model checkpoint(s)
- `predictions/client_1_test_predictions.csv`
- `predictions/client_2_test_predictions.csv`

The combined metrics table is:

- `output/track_a_seed0_summary.csv`

Generated output directories are ignored by git. Keep the CSVs/checkpoints on disk
for analysis, or force-add selected artifacts later if the paper workflow needs
committed reproducibility outputs.

## Next Work

- Run paper-quality Track A experiments over 3-5 seeds.
- Increase FL runs from the current practical `15` rounds to the guide default
  of `50` rounds, or justify the smaller value in the paper.
- Debug/tune SCAFFOLD. Current seed-0 results are unstable and much worse than
  the other methods.
- Add a script that converts `output/track_a_*_summary.csv` into a LaTeX table.
- Add plotting scripts for:
  - validation loss by round/epoch,
  - test prediction overlays,
  - communication-cost vs. RMSE Pareto points.
- Implement Track B multivariate/personalization only after Track A results are
  stable.
- Add tests for data windowing, metric inversion, communication accounting, and
  FedBN parameter exclusion.
