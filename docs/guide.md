# Federated Learning Implementation Guidelines

Reference document for implementing the federated LSTM experiment comparing centralized vs. federated active-power prediction across two submerged arc furnaces (SAFs), referred to as **Furnace 1** and **Furnace 2** throughout. This guide translates the findings from the literature review and the dataset analysis into concrete engineering decisions.

> **Anonymization constraint.** Source tag IDs (`P42.*`, `P44.*`) and the real furnace numbers (42, 44) stay strictly inside data-loading code and preprocessing configs. Nothing downstream — logs, plots, metric dumps, paper sections — should reference them. Use `furnace_1` / `furnace_2` or `client_1` / `client_2` everywhere else.

---

## 1. Research objectives

The implementation serves a single research question: **can a federated LSTM trained across two SAF clients match a centralized LSTM trained on pooled data for active-power prediction, under realistic non-IID conditions?**

Concrete deliverables the code must produce:

1. A **centralized LSTM baseline** trained on concatenated (but client-tagged) data.
2. A **local-only baseline** per client (LSTM trained on one client's data alone).
3. A **federated LSTM** trained with at least `FedAvg`, `FedProx`, `SCAFFOLD`, and `FedBN`.
4. Paired evaluation metrics on each client's held-out test window so centralized/local/federated numbers are directly comparable.
5. Communication-cost accounting (bytes transmitted per round, rounds to convergence).

Everything else — visualizations, ablations, hyperparameter sweeps — is downstream of these five artefacts.

---

## 2. Recommended repository layout

Build on the existing scaffold rather than restructuring it:

```
fl-saf/
├── dataset/
│   ├── Furnace1.csv                # client 1 raw log
│   ├── Furnace2.csv                # client 2 raw log
│   ├── README.md                   # dataset analysis report
│   └── figures/                    # analysis charts + describe CSVs
├── analyze_dataset.py              # existing EDA script (unchanged)
├── src/
│   └── fl_saf/
│       ├── __init__.py
│       ├── config.py               # dataclass-based experiment config
│       ├── data/
│       │   ├── __init__.py
│       │   ├── loader.py           # CSV → tensors, sequence windowing
│       │   ├── preprocessing.py    # scaling, constant-column filter, calibration fixes
│       │   └── splits.py           # chronological train/val/test split
│       ├── models/
│       │   ├── __init__.py
│       │   └── lstm.py             # LSTM regressor (+ FedBN variant)
│       ├── training/
│       │   ├── __init__.py
│       │   ├── centralized.py      # centralized + local-only training loop
│       │   ├── fedavg.py
│       │   ├── fedprox.py
│       │   ├── scaffold.py
│       │   ├── fedbn.py
│       │   └── utils.py            # shared training helpers, checkpointing
│       ├── evaluation/
│       │   ├── __init__.py
│       │   ├── metrics.py          # RMSE, MAE, MAPE, per-client breakdown
│       │   └── comms.py            # byte accounting for FL rounds
│       └── experiments/
│           └── run.py              # CLI entry point; picks algo by flag
├── configs/                        # YAML/TOML experiment definitions
├── experiments/                    # output: checkpoints, logs, metric CSVs
├── notebooks/                      # exploratory only, not source of truth
├── tests/
├── pyproject.toml
├── uv.lock
└── implementation_guidelines.md    # this file
```

Keep experiment outputs (metrics, checkpoints, TensorBoard logs) under `experiments/<run_id>/` with a committed `config.yaml` copy so every run is reproducible from disk alone.

---

## 3. Data realities that shape every downstream decision

The dataset analysis in `dataset/README.md` surfaces two structural facts that must be confronted **before** any modelling choice is made. Ignoring them produces misleading centralized-vs-federated comparisons.

### 3.1 Furnace 1 is a near-constant log

Only `active_power` varies on Furnace 1. All other numeric columns — voltages, currents, electrode positions/slips, hearth temperatures, gas-under-hood sensors, cooling-water channels — are constant to within ~1e-14 over the 8,150-row window. This is not a "feature skew" problem; it is an information-content problem. Furnace 1 contributes essentially one informative signal plus a timestamp.

**Consequences:**

- A multivariate LSTM with the full 19-column shared schema will degenerate on Furnace 1 to a univariate model plus constant noise features — the gradient flow through every non-`active_power` input will be zero.
- Local normalization (z-scoring) breaks on constant columns (zero variance → division by zero). This must be handled explicitly.
- Any claim that "FL recovers centralized performance on Furnace 1" will be trivially true if Furnace 1's task is near-degenerate. The experimental design must not over-credit FL for easy wins.

### 3.2 Furnace 2 has calibration and near-constant channels

- `voltage_c` sits ~288 while `voltage_a`/`voltage_b` sit ~17 — likely a unit/calibration issue, not physical phase imbalance.
- `current_a` is two orders of magnitude larger than `voltage_a` — again a calibration/scale anomaly.
- `current_b`, `current_c`, `electrode_slip_b`, `electrode_slip_c`, `water_after_cooling_2`, and `cos_phi` have `std < 0.002` — effectively stuck.
- `cos_phi` reports physically implausible values (~−20), indicating the raw tag is not a proper power-factor value.

### 3.3 Modelling decision: run two experiment tracks

Rather than papering over the asymmetry, run **two parallel experiments**:

**Track A — Univariate active-power forecasting (primary).**
Both clients use only `active_power` as input and target (sequence-to-one forecasting with a horizon *h*). This is the cleanest, most defensible comparison: both clients genuinely have this signal and it genuinely varies. FedAvg/FedProx/SCAFFOLD/FedBN results here go in the main paper.

**Track B — Multivariate with shared schema + personalization (secondary).**
Both clients use the shared 19-column schema. Before training, per-client filter drops columns with `std < 1e-3` — this will keep ~15 columns for Furnace 2 and collapse Furnace 1 to effectively active_power only. A personalized / FedBN approach is the natural fit here since feature availability is asymmetric. Track B results are reported as an ablation showing FL's behaviour under extreme feature skew, not as the headline comparison.

The literature review supports this split: FedST's "dual spatio-temporal feature skew" (Chen et al., ACM MM 2024) and AttFL's attention-based feature-map exchange (ACM IMWUT 2023) both argue for feature-aware personalization exactly when clients have heterogeneous sensor availability.

Do **not** include Furnace 2's control-loop signals (`voltage_step_*`, `power_setpoint`, `cos_phi`) in the shared global model. They have no equivalent on Furnace 1 and — per the tag mapping notes — belong in a local personalization head if used at all.

---

## 4. Data preprocessing pipeline

Implement as a pure `preprocess_client(df, config) -> tuple[X, y, scaler, feature_names]` function with no hidden state. A single ordered pipeline:

### 4.1 Load and align

```python
df = pd.read_csv(path, parse_dates=["DateTime"]).sort_values("DateTime").reset_index(drop=True)
```

Verify the 60 s cadence (`df["DateTime"].diff().value_counts()`) and fail fast if resampling is needed — the analysis confirms both clients are already on a uniform minute grid.

### 4.2 Drop constant / near-constant columns (Track B only; skipped for Track A)

```python
drop_cols = [c for c in numeric_cols if df[c].std() < EPS]  # EPS = 1e-3
df = df.drop(columns=drop_cols)
```

Log the dropped set per client — it's a directly reportable non-IID metric in the paper.

### 4.3 Calibration sanity checks (Track B only)

For Furnace 2, log warnings (do not auto-correct) when:

- `voltage_c.mean()` differs from `(voltage_a.mean() + voltage_b.mean()) / 2` by more than an order of magnitude.
- `current_a.mean()` differs from `current_b.mean()` by more than an order of magnitude.
- `cos_phi` falls outside `[-1, 1]`.

Flag these in the preprocessing report but keep the raw values; the paper should acknowledge the calibration issue honestly rather than hiding it.

### 4.4 Chronological split

Because this is a forecasting problem, random shuffling leaks future into past. Split chronologically per client: **70% train / 15% validation / 15% test** on the time axis, with sequences never crossing split boundaries.

### 4.5 Per-client standardization

Fit a `StandardScaler` on the **train portion only** of each client's data. The federated BN / FedBN literature (Li et al., ICLR 2021) motivates keeping normalization statistics local rather than sharing them — this is consistent with that philosophy and makes Track A/Track B comparable.

For Track A (univariate) you still z-score `active_power` per client; inverse-transform before computing reportable MW-scale metrics.

### 4.6 Sequence windowing

Sliding window with:

- `seq_len`: 60 (i.e. 1 hour of context at 60 s cadence) as the default; run `{30, 60, 120}` as an ablation.
- `horizon`: 1 (predict active power at t+1) as the default; also run `{5, 15}` to demonstrate multi-step behaviour.
- `stride`: 1 for training, 1 for eval.

Produce `X` of shape `(N, seq_len, n_features)` and `y` of shape `(N, horizon)` as `float32` tensors. Use PyTorch `TensorDataset` + `DataLoader`.

---

## 5. LSTM model architecture

Single module, parameterized by config, shared between centralized and federated training. Follow the smart-building anomaly-detection benchmark from the literature review (stacked LSTM outperformed GRU on that task) and keep the model small enough that federated communication stays tractable.

```python
class LSTMRegressor(nn.Module):
    def __init__(
        self,
        input_size: int,
        hidden_size: int = 64,
        num_layers: int = 2,
        dropout: float = 0.2,
        horizon: int = 1,
        use_batchnorm: bool = False,  # True for FedBN track
    ):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.bn = nn.BatchNorm1d(hidden_size) if use_batchnorm else nn.Identity()
        self.head = nn.Linear(hidden_size, horizon)

    def forward(self, x):  # x: (B, T, F)
        out, _ = self.lstm(x)
        last = out[:, -1, :]  # (B, H)
        return self.head(self.bn(last))  # (B, horizon)
```

Defaults to start with: `hidden_size=64`, `num_layers=2`, `dropout=0.2`. These are deliberately small so per-round communication in FL stays on the order of a few MB, matching the compression-free baseline discussed in the literature review (the 28.6 M-parameter figure in Xu et al. 2023 is a cautionary tale, not a target).

Ablate `{hidden_size ∈ {32, 64, 128}, num_layers ∈ {1, 2, 3}}` only after the main experiments are stable.

---

## 6. Centralized and local-only baselines

Both are non-negotiable reference points for the paper's main claim and must be implemented first — before any FL code.

**Centralized.** Concatenate per-client training sequences after per-client standardization (the scalers stay client-specific even in centralized training — otherwise Furnace 2's scale dominates the loss and the centralized baseline becomes artificially weak, which would flatter FL unfairly). Train one LSTM on the pooled dataset.

**Local-only.** Train one LSTM per client, on that client's training set only, with identical hyperparameters.

Training loop essentials:

- **Loss:** MSE in standardized space during training; RMSE/MAE reported in the original MW scale after inverse-transform.
- **Optimizer:** Adam with `lr=1e-3`, `weight_decay=1e-5`. Cosine or step LR schedule.
- **Batch size:** 64.
- **Epochs:** up to 100 with early stopping on validation loss (patience 10).
- **Clipping:** gradient-norm clip at 1.0 (LSTMs benefit consistently).
- **Seeds:** fix `torch`, `numpy`, `random` seeds; run each configuration with 3–5 seeds and report mean ± std.

Save predictions (not just metrics) per run; reuse them in the paper's figures so centralized/local/federated curves are drawn from identical evaluation code.

---

## 7. Federated learning implementation

### 7.1 Framework choice

Two defensible paths:

- **Flower (`flwr`)** — mature FL framework, supports FedAvg/FedProx out of the box, easy to add custom strategies. Preferred if you want to minimize glue code.
- **From-scratch FL loop** — a ~200-line orchestrator that holds client states, runs local training, aggregates. Preferred if you want full transparency of the communication accounting (which is a paper deliverable) and tight control over SCAFFOLD/FedBN internals.

Recommendation: **implement from scratch**. With only two clients and four algorithms, a custom loop is simpler than adapting Flower's strategies for SCAFFOLD/FedBN, and the communication-cost accounting is much more auditable. Write it so it could be swapped for Flower later if the client count grows.

Core loop sketch:

```python
def federated_train(clients, global_model, strategy, cfg):
    for r in range(cfg.n_rounds):
        round_updates = []
        for c in clients:
            c.set_weights(strategy.prepare_client_state(global_model, c))
            local_stats = c.local_train(cfg.local_epochs, cfg.local_lr)
            round_updates.append(strategy.extract_update(c, local_stats))
        global_model = strategy.aggregate(global_model, round_updates)
        log_round_metrics(r, clients, global_model)
```

Each strategy (`FedAvg`, `FedProx`, `SCAFFOLD`, `FedBN`) is a class with `prepare_client_state`, `extract_update`, `aggregate`.

### 7.2 Algorithm specifics

**FedAvg** (McMahan et al., AISTATS 2017). Sample-weighted average of client weights. With 2 clients and similar dataset sizes this collapses to simple averaging. Serves as the floor.

**FedProx** (Li et al., MLSys 2020). Adds a proximal term `(μ/2) * ||w - w_global||²` to each client's local loss. Ablate `μ ∈ {0.001, 0.01, 0.1, 1.0}`. Literature review flag: with small `μ`, FedProx tracks FedAvg closely — expect modest gains.

**SCAFFOLD** (Karimireddy et al., ICML 2020). Maintains client and server control variates to correct client drift. Per-client state `c_i` must persist between rounds — store on the client object, not the server. Update rule for local step: `w ← w − η·(∇f_i(w) − c_i + c)`. The literature review (Prathusha & Aparna 2025) cites SCAFFOLD as the most accurate on HAR-style data but also the most communication-heavy per round (~2× FedAvg because control variates are transmitted alongside weights); log this honestly in the comms table.

**FedBN** (Li et al., ICLR 2021). Average all parameters **except** BatchNorm layers; keep BN stats per client. Requires the LSTM variant with `use_batchnorm=True`. Exclude BN `weight`, `bias`, `running_mean`, `running_var` from the aggregation set. This is the most directly motivated algorithm for this dataset given the calibration heterogeneity on Furnace 2.

### 7.3 FL hyperparameters

Start with:

- `n_rounds`: 50 (enough for FedAvg to plateau on simple tasks per literature; extend to 100 if still improving).
- `local_epochs`: 2 (small; more local work amplifies client drift under non-IID).
- `local_lr`: 1e-3.
- `batch_size`: 64.

Total local work per round = 2 clients × 2 epochs × (training set / batch size) gradient steps.

### 7.4 Communication accounting

Per round, log:

- Number of float32 parameters transmitted per client upload.
- Number of float32 parameters transmitted per server download (same model → symmetric for FedAvg, asymmetric for SCAFFOLD because control variates are added).
- Total bytes per round and cumulative across training.
- Round at which validation RMSE first reaches a reference threshold (e.g., centralized baseline + 10%), for comparability with the literature review's "rounds to convergence" numbers.

---

## 8. Evaluation protocol

Consistency across centralized/local/federated is the single most important correctness property.

### 8.1 Metrics

Report in inverse-transformed (MW) scale, per client, on the held-out chronological test window:

- **RMSE** — primary headline metric, aligns with RUL literature conventions.
- **MAE** — robust to outliers, easier to interpret.
- **MAPE** — useful if `active_power` never approaches zero in the test window; verify this before including.
- **R²** — communicates how much variance the model explains; helpful when MW-scale numbers are hard to intuit.

### 8.2 Reporting structure

Produce a table per algorithm with this shape:

| Algorithm | Client | Test RMSE (MW) | Test MAE (MW) | Test R² | Comm. rounds | Comm. MB |
|---|---|---|---|---|---|---|
| Centralized | 1 | … | … | … | — | — |
| Centralized | 2 | … | … | … | — | — |
| Local only | 1 | … | … | … | — | — |
| Local only | 2 | … | … | … | — | — |
| FedAvg | 1 | … | … | … | R | M |
| FedAvg | 2 | … | … | … | R | M |
| FedProx | … | | | | | |
| SCAFFOLD | … | | | | | |
| FedBN | … | | | | | |

Every cell is mean ± std over 3–5 seeds.

### 8.3 Figures for the paper

- Training/validation loss vs. round for each FL algorithm, both clients overlaid.
- Test-set prediction overlay (ground truth vs. prediction) for a representative 12-hour window per client, one panel per algorithm.
- Communication cost vs. RMSE Pareto plot: one point per (algorithm, seed), showing that e.g. FedBN reaches lower RMSE at comparable comm cost to FedAvg.

---

## 9. Configuration and reproducibility

- One `ExperimentConfig` dataclass drives everything. Dump it to YAML at run start and load it back at analysis time.
- CLI: `python -m fl_saf.experiments.run --algo fedbn --config configs/fedbn_default.yaml --seed 0`.
- Deterministic mode: `torch.use_deterministic_algorithms(True)` plus `CUBLAS_WORKSPACE_CONFIG=:4096:8`; accept the small speed cost.
- Log to both a per-run CSV (for programmatic analysis) and TensorBoard / Weights & Biases (for inspection). Never rely on stdout alone.
- Pin every dependency in `pyproject.toml`; `uv.lock` is the source of truth.

---

## 10. Suggested implementation phases

Each phase is an end-to-end working system, not an intermediate state:

1. **Phase 1 — Data + centralized baseline.** Preprocessing pipeline, sequence windowing, centralized LSTM, metrics, at least one end-to-end training run on Track A. Exit criterion: centralized RMSE numbers per client that you would be willing to quote in the paper.
2. **Phase 2 — Local-only baselines.** Same model, per-client training, same metrics pipeline. Exit criterion: local-only numbers produced and sanity-checked (Furnace 1 local should be competitive because Track A is univariate).
3. **Phase 3 — FedAvg.** Minimal FL loop, communication accounting. Exit criterion: FedAvg matches or beats the weaker local-only baseline on both clients.
4. **Phase 4 — FedProx + SCAFFOLD + FedBN.** Swap strategies behind a single interface. Exit criterion: all four algorithms reporting into the same metrics table.
5. **Phase 5 — Track B (multivariate + personalization).** Only after Track A is fully written up.
6. **Phase 6 — Ablations.** Sequence length, horizon, hidden size, local epochs, `μ` for FedProx.

Do not start Phase 3 before Phases 1–2 are producing clean, reproducible numbers. Most FL debugging pain comes from unstable baselines, not from the FL code itself.

---

## 11. Hard constraints

Non-negotiable rules for the codebase and anything derived from it:

1. **No real tag IDs or furnace numbers outside `src/fl_saf/data/`.** CSV column names must already be anonymized at load time.
2. **No fabricated metrics.** If a number is not produced by the code on committed data, it does not appear in the paper.
3. **No silent data mutation.** Every preprocessing decision (constant-column drop, calibration flag, scaler fit) is logged with the run.
4. **No train/test leakage.** Chronological splits only; scalers fit on train only; sequence windows never cross split boundaries.
5. **No unreported communication.** Every byte transmitted in FL is accounted for in the comms table.
6. **No comparison against a handicapped centralized baseline.** Centralized must use per-client standardization (same as FL) so the comparison is about the aggregation mechanism, not the normalization asymmetry.
7. **Seeds are logged, runs are repeatable.** A reader should be able to reproduce any table cell from the committed config + seed.

---

## 12. Open questions to resolve before Phase 3

These are decisions the researcher (not Claude Code) should make, because they have paper-framing implications:

- **Track A as the headline, Track B as ablation — confirm?** This guide assumes yes. The alternative is to lead with Track B and explicitly frame Furnace 1's degeneracy as part of the non-IID stress test.
- **Prediction horizon.** 1-step (60 s ahead) is the conservative default; 5- or 15-step-ahead forecasting is more operationally meaningful but harder and will widen the FL/centralized gap.
- **Do we report intermediate FL checkpoints as separate models?** Literature often reports "best validation round" which implicitly uses a validation signal the federated setting may not realistically have. Consider reporting both "best-val" and "final-round" numbers.
- **Differential privacy.** The literature review covers DP extensively and the project context emphasizes privacy motivation. DP is not in this guide's scope but decide early whether a DP ablation belongs in the paper — adding it later means rerunning every FL experiment.
