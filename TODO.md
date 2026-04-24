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

Track A seed-0, seed-1, and seed-2 runs were executed and saved under
`output/` using run IDs of the form:

- `output/track_a_seed<seed>_centralized/`
- `output/track_a_seed<seed>_local_only/`
- `output/track_a_seed<seed>_fedavg/`
- `output/track_a_seed<seed>_fedprox/`
- `output/track_a_seed<seed>_scaffold/`
- `output/track_a_seed<seed>_fedbn/`

Each run contains:

- `config.yaml`
- `preprocessing.json`
- `history.csv`
- `metrics.csv`
- model checkpoint(s)
- `predictions/client_1_test_predictions.csv`
- `predictions/client_2_test_predictions.csv`

The combined metrics tables are:

- `output/track_a_summary.csv` - one row per algorithm/client/seed.
- `output/track_a_summary_stats.csv` - mean/std per algorithm/client.

Generated output directories are ignored by git. Keep the CSVs/checkpoints on
disk for analysis, or force-add selected artifacts later if the paper workflow
needs committed reproducibility outputs. The summary CSVs are small enough to
commit.

## Decisions

- **Extend Track A to 5 seeds (add seeds 3 and 4).** At 3 seeds, FedAvg,
  FedProx, centralized, and local-only baselines are already stable (RMSE
  coefficient of variation < 3% in most cells), but FedBN's client-2 RMSE
  spreads across 1.212 / 1.230 / 1.506 (CV ~12.5%) and its client-1 CV is
  ~7.8%. Since the guide frames FedBN as the most directly motivated
  algorithm for this dataset's calibration heterogeneity, the paper's
  headline claim about it is under-powered at n=3 and needs tighter
  confidence intervals. FedAvg/FedProx/centralized/local_only benefit
  "for free" from the same reruns. SCAFFOLD should be debugged before
  spending additional seeds on it — its current divergence is not a
  variance problem.

## Next Work

- Run Track A seeds 3 and 4 for all algorithms (or skip SCAFFOLD until
  the divergence is fixed). Regenerate `track_a_summary.csv` and
  `track_a_summary_stats.csv` with n_seeds=5.
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
