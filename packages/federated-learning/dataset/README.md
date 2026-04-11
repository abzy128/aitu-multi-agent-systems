# Furnace Dataset — Analysis Report

This directory contains two CSV logs, `Furnace1.csv` and `Furnace2.csv`, that
serve as the two clients for the federated learning experiment in this
package. Each file is a minute-by-minute dump of process signals from an
industrial electric arc / ore-thermal furnace.

The analysis below is produced by [`analyze_dataset.py`](../analyze_dataset.py).
Run it from the package root:

```bash
uv run python analyze_dataset.py
```

Charts are written into [`figures/`](./figures/) and per-feature summary
statistics into `figures/Furnace{1,2}_describe.csv`.

---

## 1. Dataset at a glance

| Property            | Furnace1                | Furnace2                |
|---------------------|-------------------------|-------------------------|
| Rows                | 8 150                   | 8 150                   |
| Columns             | 27 (26 numeric + time)  | 25 (24 numeric + time)  |
| Time range          | 2026-03-25 → 2026-03-30 | 2026-03-25 → 2026-03-30 |
| Duration            | ~144 h (6 days)         | ~144 h (6 days)         |
| Sampling period     | 60 s                    | 60 s                    |
| Missing values      | 0                       | 0                       |
| File size           | 4.0 MB                  | 3.8 MB                  |

Both files share the same temporal window and cadence, which makes them
directly alignable on `DateTime` — useful when we later want paired training
runs or common evaluation windows.

---

## 2. Features

### 2.1 Shared schema

The two furnaces expose a common electrical / mechanical core:

- **Power:** `active_power`, `reactive_power`
- **3-phase voltage:** `voltage_a`, `voltage_b`, `voltage_c`
- **3-phase current:** `current_a`, `current_b`, `current_c`
- **Electrode geometry:** `electrode_position_{a,b,c}`, `electrode_slip_{a,b,c}`
- **Hearth temperature:** `hearth_temp_1`
- **Cooling water:** `water_temp_to_furnace_{1,2}`, `water_after_cooling_{1,2}`

### 2.2 Client-specific features

Furnace1 adds the **off-gas / hearth / hydraulic sensors**:

- `hearth_temp_2`, `hearth_temp_3`
- `gas_under_hood_1`, `gas_under_hood_2`, `gas_under_hood_3`
- `water_pressure_to_furnace_1`, `water_pressure_to_furnace_2`

Furnace2 adds the **control-loop signals**:

- `voltage_step_a`, `voltage_step_b`, `voltage_step_c` (tap-changer steps)
- `power_setpoint` (target active power)
- `cos_phi` (power factor)

![Feature overlap](figures/feature_overlap.png)

This schema mismatch is deliberate: it mimics the realistic federated setting
where each client monitors its own equipment with its own sensor suite. Any FL
training loop has to either (a) restrict to the shared columns, or (b) pad
missing columns per client.

---

## 3. Data quality findings

Running `analyze_dataset.py` surfaces two structural issues that dominate any
modelling decision downstream.

### 3.1 Furnace1 is a near-constant log

For Furnace1, **only `active_power` actually varies**; every other numeric
column holds a single value for the entire 8 150-row window (`std ≈ 0`, or
exactly zero for `reactive_power`). Concretely:

- `reactive_power` is identically 0.
- All three phase voltages sit at ~14.35.
- All three phase currents sit at ~17.85.
- Electrode positions/slips, hearth temps, under-hood gas readings, and all
  water loop signals are constant to ~1e-14 tolerance.

This is visible both in `Furnace1_describe.csv` and in the distribution /
correlation charts (the correlation heatmap is effectively empty because
constant columns are dropped before correlation is computed).

**Implication for federated learning:** Furnace1 contributes essentially one
informative signal — `active_power` — plus its time index. Training on it
locally is only meaningful for unary/temporal models of active power.

### 3.2 Furnace2 has several flat and a few out-of-range channels

Furnace2 is much richer but not clean:

- `current_b`, `current_c`, `electrode_slip_b`, `electrode_slip_c`,
  `water_after_cooling_2`, `cos_phi` all have `std < 0.002` — effectively
  stuck at one value (and some are physically implausible, e.g. negative
  `cos_phi ≈ -20`).
- `voltage_c` sits around 288 while `voltage_a/b` sit near 17 — the `c`
  channel is on a different scale, possibly mis-labelled or miscalibrated.
- `current_a` is two orders of magnitude larger than `voltage_a`, again
  consistent with a calibration / unit issue.
- `electrode_position_b/c` and `electrode_slip_a` are noisy but bounded, and
  likely the strongest non-power signals.

None of these are "missing values" in the `NaN` sense, so trivial
`df.isna()` checks will pass. Any preprocessing step has to explicitly
filter constant / near-constant columns and rescale the suspicious channels.

---

## 4. Feature analysis charts

All charts below are produced by `analyze_dataset.py` and live under
`figures/`.

### 4.1 Distributions

![Furnace1 feature distributions](figures/Furnace1_distributions.png)
![Furnace2 feature distributions](figures/Furnace2_distributions.png)

Furnace1's plot effectively contains a single histogram (for `active_power`);
the rest are Dirac spikes. Furnace2 shows multi-modal behaviour in
`active_power`, `voltage_c`, `current_a`, and the electrode positions, which
is characteristic of an alternating "idle vs. operating" regime.

### 4.2 Correlation structure

![Furnace2 correlation matrix](figures/Furnace2_correlation.png)

On Furnace2, `active_power`, `voltage_c`, `current_a`, and `power_setpoint`
form the obvious correlated block — the electrical load is tracking the
setpoint. Electrode positions correlate with each other but not strongly with
the electrical group, so they carry independent information. (A Furnace1
correlation matrix is not meaningful because all non-target columns are
constant and get dropped.)

### 4.3 Key signals over time

![Furnace1 time series](figures/Furnace1_timeseries.png)
![Furnace2 time series](figures/Furnace2_timeseries.png)

Furnace1 shows the `active_power` trajectory only. Furnace2 exposes both the
electrical load and process-side dynamics.

### 4.4 Engineered features

`analyze_dataset.py` derives a few physically meaningful features before
plotting:

- `voltage_mean` / `voltage_imbalance` — mean and std across the three phase
  voltages (phase-asymmetry proxy).
- `current_mean` / `current_imbalance` — same for phase currents.
- `apparent_power = √(active² + reactive²)` and
  `derived_cos_phi = active / apparent`.
- `active_power_roll5` — 5-minute rolling mean of active power.
- Temporal bucketing: `hour`, `minute_of_day`.

![Furnace2 engineered features](figures/Furnace2_engineered.png)

For Furnace1, the engineered features collapse because their inputs are
constant; for Furnace2 they give a compact view of how balanced the phases
are and how the power factor behaves.

### 4.5 PCA scree

![Furnace2 PCA scree](figures/Furnace2_pca.png)

On Furnace2, the first two principal components already capture the bulk of
the variance, reflecting the strong correlation block between load,
setpoint and current. Furnace1 has < 2 varying numeric columns, so PCA is
not run for it.

---

## 5. Implications for the federated learning experiment

1. **Non-IID by construction.** Furnace1 and Furnace2 differ in schema,
   variance structure, and value ranges. Any naive `FedAvg` over raw
   features will be dominated by Furnace2. This dataset is therefore a
   realistic stress test for robust aggregation (e.g. `FedProx`,
   per-client normalization, or feature-wise masking).

2. **Use the shared schema for the global model.** The 19 shared columns
   (listed in §2.1) are the natural intersection for a common model.
   Client-specific features can still be used by local heads.

3. **Preprocess before training, not after.** Always drop constant /
   near-constant columns per client (`std < ε`) and apply per-client
   standardization — otherwise Furnace1's degenerate columns will inject
   zeros into the gradient and Furnace2's scale mismatch on `voltage_c` /
   `current_a` will dominate the loss.

4. **Target candidates.** `active_power` is the only signal both clients
   agree on and actually varies on both sides, which makes it the default
   target for a first joint forecasting / anomaly-detection experiment.
   `power_setpoint` on Furnace2 can act as an auxiliary label locally.

5. **Time alignment is free.** Identical 60-second cadence and overlapping
   windows mean no resampling is needed before federated rounds.

---

## 6. Files in this directory

| File                             | Description                                        |
|----------------------------------|----------------------------------------------------|
| `Furnace1.csv`                   | Client 1 raw log                                   |
| `Furnace2.csv`                   | Client 2 raw log                                   |
| `README.md`                      | This report                                        |
| `figures/Furnace{1,2}_describe.csv` | Per-feature summary statistics                  |
| `figures/*.png`                  | Charts referenced above                            |
| `figures/summary.json`           | Machine-readable dataset summary                   |
