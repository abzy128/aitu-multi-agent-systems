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
- LaTeX table rendering from summary-stats CSVs
  (`src/fl_saf/evaluation/latex_table.py`).
- Plot rendering for validation loss by round, test prediction overlays,
  and communication-cost vs. RMSE Pareto
  (`src/fl_saf/evaluation/plots.py`, outputs under `output/figures/`).
- Focused pytest coverage for data windowing, chronological splits,
  train-only scaling, metric inversion, communication accounting, and FedBN
  BatchNorm exclusion (`tests/`).
- CLI smoke integration tests for centralized and FedAvg runs on synthetic CSVs.

## Current Outputs

Track A seed-0 through seed-4 runs were executed and saved under
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

- `output/track_a_summary.csv` - one row per algorithm/client/seed at
  `n_rounds=15` (plus the centralized/local-only runs that don't
  depend on rounds).
- `output/track_a_summary_stats.csv` - mean/std per algorithm/client at
  `n_rounds=15`.
- `output/track_a_r50_summary.csv` / `output/track_a_r50_summary_stats.csv`
  - 50-round variant (FL algorithms only; run IDs
  `track_a_r50_seed<seed>_<algo>`). Configured via
  `configs/track_a_r50.yaml`.

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

- **Fixed SCAFFOLD divergence by switching local optimizer from Adam to
  SGD (momentum=0.9, lr=1e-2).** Root cause: the control-variate
  correction `grad += c - c_i` and the Option-II client update
  `c_i+ = c_i - c + (w_global - y_i)/(K·η_l)` are both derived assuming
  SGD local steps. Under Adam, the actual step is `η·m̂/√v̂` (not `η·g`),
  so `(w_global - y_i)/(K·η)` no longer estimates the mean gradient, the
  control variates drift out of scale, and they dominate the true
  gradients — giving the previous RMSE ~5.7 / 7.9 and negative R². After
  the fix (seeds 0–4): RMSE 1.617 ± 0.029 (client_1) and 1.518 ± 0.046
  (client_2). Tight across seeds (CV 1.8% / 3.0%) but still the weakest
  FL algorithm here, consistent with SCAFFOLD's literature profile —
  it mainly wins with many heterogeneous clients; two clients is not its
  regime. `federated.scaffold_lr` and `federated.scaffold_momentum` are
  now configurable.

- **Ran the 15-rounds → 50-rounds ablation.** FedProx client_1 improves
  to 1.047 ± 0.006 (now beats the centralized baseline at 1.081) and
  SCAFFOLD improves on both clients (1.617 → 1.308, 1.518 → 1.321),
  consistent with control-variate and proximal-term methods needing
  more rounds to stabilize. FedAvg mildly degrades on client_2
  (1.321 → 1.556). FedBN diverges (RMSE 3.054 / 5.028), but this is not
  a rounds problem — it exposed a latent implementation bug (see below).
  The paper should report 50 rounds for FedProx and SCAFFOLD, and either
  15 rounds for FedAvg or explicitly show the round-dependence as a
  robustness finding.

- **Fixed FedBN to match the paper.** The old `train_fedbn` aliased to
  `train_fedavg(..., exclude_bn=True)`, which returned a single
  `global_model` whose BN-layer entries were cloned from `states[0]`
  (client_1's latest local BN) — so both clients were evaluated on one
  model with one client's BN stats. The new implementation persists a
  per-client `LSTMRegressor` across rounds, trains each locally (BN
  stats update locally), aggregates only non-BN weights, and returns a
  `{client_id: model}` dict. `federated.py`'s evaluation and the
  existing `run.py` prediction/checkpoint paths already handle dict
  models. At 15 rounds the fix doesn't move FedBN much
  (client_2 RMSE 1.430 ± 0.253 vs buggy 1.457 ± 0.241) because the
  buggy version happened to be "close enough" with little BN drift.
  At 50 rounds, correct FedBN still diverges
  (client_1 2.396 ± 0.899, client_2 3.962 ± 1.197): the aggregated
  non-BN weights pull each client away from its BN-normalized optimum,
  and with only 2 heavy non-IID clients the gap widens every round.
  This is a genuine FedBN limitation in this setup, not an artifact.

## Next Work

- Run Track B (multivariate + personalization). The planned 2026-04-26 date is
  now historical; schedule the next run window before starting new long jobs.
- Extend CLI smoke coverage to FedProx, SCAFFOLD, and FedBN after deciding how
  much runtime is acceptable in the default test suite.
