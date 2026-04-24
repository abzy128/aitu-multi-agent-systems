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

Track A seed-0 through seed-4 runs were executed and saved under
`output/` using run IDs of the form (SCAFFOLD only has seeds 0–2 until
the divergence is fixed):

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

- **Extended Track A to 5 seeds (seeds 3 and 4 added, SCAFFOLD skipped).**
  Rationale: at 3 seeds, FedAvg/FedProx/centralized/local-only RMSE CV was
  < 3% in most cells, but FedBN's client-2 CV was ~12.5% (RMSE spread
  1.212 / 1.230 / 1.506). Since the guide frames FedBN as the algorithm
  most directly motivated for this dataset's calibration heterogeneity,
  n=3 was under-powered for the headline claim. SCAFFOLD was skipped
  because its current divergence is not a variance problem — debug first.
  Outcome: the extension widened FedBN's variance rather than tightening
  it. Client-2 RMSE moved to 1.457 ± 0.241 (CV 16.5%) and client-1 to
  1.259 ± 0.119 at n=5 — seeds 3 and 4 produced the two worst FedBN runs,
  confirming genuine instability rather than noisy sampling. FedProx is
  now the tightest FL result (CV < 1.5% both clients). This is a
  scientifically useful finding, not a setback.

## Next Work

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
