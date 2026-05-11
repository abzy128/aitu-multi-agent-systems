# Claude Code Guidelines — LaTeX Paper: Federated Learning for LSTM on SAF Sensor Data

## Project Overview

You are writing sections of a research paper that applies **federated learning (FL)** to **LSTM-based models** trained on industrial sensor data from two **Submerged Arc Furnaces (SAFs)** at a manufacturing facility in Kazakhstan. The core contribution is demonstrating FL as a privacy-preserving alternative to centralized LSTM training, where each furnace acts as a separate FL client.

**The `.bib` file is already populated and up to date — do not create or modify it.** Use `\cite{}` commands referencing existing BibTeX keys. If you are unsure of an exact key, leave a `\cite{TODO:description}` placeholder and flag it with a `% TODO` comment.

---

## Sections to Write

### 1. Introduction

Write a complete Introduction section covering the following narrative arc:

1. **Industrial context.** SAF operations in ferroalloy/metallurgical production generate high-frequency multivariate sensor data (electrical parameters, electrode positions, thermal readings, cooling system telemetry). Continuous monitoring and predictive modelling of active power and process health is critical for operational efficiency and safety.

2. **LSTM suitability.** LSTM networks are well-suited for this time-series data due to their ability to capture long-range temporal dependencies in sequential sensor readings — electrode positions, currents, voltages, and temperatures evolve with complex temporal dynamics.

3. **The data privacy and heterogeneity problem.** Centralizing raw sensor data from multiple furnaces raises privacy, proprietary, and regulatory concerns. Additionally, furnaces differ in instrumentation density (Furnace 42 has richer thermal/gas sensors; Furnace 44 has unique voltage-step and power-setpoint tags), operating regimes, and maintenance histories — creating natural data heterogeneity.

4. **Federated learning as the solution.** FL enables collaborative model training without raw data exchange. Each furnace trains locally and shares only model updates. This preserves data privacy while leveraging cross-furnace knowledge.

5. **Literature gap.** While FL has been applied to turbofan RUL prediction (C-MAPSS), rotating machinery fault diagnosis, smart grids, and IoT anomaly detection, **no prior work applies FL to SAF operational modelling**. The SAF setting introduces unique challenges: extreme inter-furnace sensor heterogeneity (mismatched tag sets), high-dimensional electrical/thermal feature spaces, and the need for real-time power prediction under non-IID temporal distributions.

6. **Research objectives.** State clearly:
   - Compare centralized vs. federated LSTM training for active power prediction across two SAFs.
   - Evaluate the impact of inter-furnace data heterogeneity on FL convergence and accuracy.
   - Assess communication efficiency and practical feasibility of FL in this industrial setting.

7. **Paper structure.** Brief paragraph outlining remaining sections (Literature Review, Methodology, Experiments, Results, Discussion, Conclusion).

**Tone and style:** Academic, third-person, concise. Avoid filler phrases. Every paragraph should advance the argument.

**Key references to cite** (use whatever BibTeX keys exist in the `.bib` file for these):
- McMahan et al., 2017 (FedAvg, original FL paper)
- Li et al., 2020 (FedProx, convergence theory)
- Karimireddy et al., 2020 (SCAFFOLD)
- Kamei & Taghipour, 2023 (centralized vs. federated RUL comparison)
- Liu et al., 2021 (AMCNN-LSTM for IIoT anomaly detection)
- Li et al., 2021 (FedBN — batch normalization for feature shift)
- McMahan et al., 2018 (user-level DP for federated LSTM)
- Chen et al., 2025 (FeDaL — temporal resolution bias)
- FedST, 2024 (dual spatio-temporal skew)
- Yang et al., 2025 (FL overhead ~0.6% for LSTM)
- Any SAF/ferroalloy domain references if present in `.bib`

---

### 2. Methodology (Provisional)

Write a provisional Methodology section. Mark any subsection that depends on unfinished experimental work with `% PROVISIONAL — to be updated with final experimental details`. Structure it as follows:

#### 2.1 Data Description

- **Source:** Time-series sensor data from two SAFs (referred to as Furnace 1 and Furnace 2 in the paper — **never use real tag prefixes or facility-identifying information in the paper text**).
- **Target variable:** Active power (one tag per furnace).
- **Feature groups** (describe generically without revealing tag IDs):
  - Electrical parameters: reactive power, phase voltages (A/B/C), electrode currents (A/B/C).
  - Electrode control: electrode positions (A/B/C), electrode slip (A/B/C).
  - Thermal — furnace body: hearth temperatures, gas-under-hood temperatures (Furnace 1 only).
  - Cooling system: water inlet temperatures, water post-cooling temperatures, water pressures (partial availability).
- **Sensor heterogeneity:** Explicitly state that Furnace 1 has additional thermal sensors (3 hearth temperature points, 3 gas-under-hood points) not present on Furnace 2. Furnace 2 has exclusive features (voltage step per phase, power setpoint, cos φ) absent on Furnace 1.
- **Shared feature set for FL:** Only features available on both furnaces enter the global FL model. Furnace-exclusive features may be used in local personalization layers. For thermal alignment, only hearth temperature point 1 from Furnace 1 is used.
- **Preprocessing:** Mention retrieval mode (`Average` for analog signals), handling of nulls/missing values, normalization strategy (min-max or z-score — mark provisional), and sliding window construction for LSTM input sequences.

#### 2.2 LSTM Architecture

- Describe the LSTM model architecture to be used: input layer dimensionality (= number of shared features), number of LSTM layers, hidden units per layer, dropout, output layer for active power regression.
- Mark hyperparameters as provisional placeholders, e.g.:
  ```latex
  % PROVISIONAL — hyperparameters to be finalized after tuning
  The model consists of \texttt{[N]} stacked LSTM layers with \texttt{[H]} hidden units each...
  ```
- Briefly justify LSTM over alternatives (GRU, Transformer) by citing:
  - Communication cost advantage: LSTM vs. Transformer convergence in FL (FL-HDECOC, 2025).
  - GRU has fewer parameters but federated stacked LSTMs consistently outperform federated GRUs in anomaly detection benchmarks.

#### 2.3 Federated Learning Setup

- **FL framework:** Each furnace is one FL client. A central server orchestrates aggregation. Describe the FedAvg baseline algorithm step by step.
- **Communication protocol:** Each round consists of: (1) server broadcasts global model, (2) each client performs E local epochs of SGD on local data, (3) clients send updated model weights to server, (4) server aggregates via weighted averaging.
- **Aggregation strategies to compare:** List FedAvg, FedProx (with proximal term μ), and optionally SCAFFOLD or an adaptive optimizer variant (e.g., Mime). Mark which will actually be implemented — provisional.
- **Key FL hyperparameters** (all provisional):
  - Number of communication rounds
  - Local epochs per round (E)
  - Local batch size
  - Learning rate and schedule
  - FedProx proximal term μ
- **Handling heterogeneity:** Describe approach to non-IID data:
  - Shared feature alignment (only common features in global model).
  - Optional personalization layers for furnace-specific features.
  - Potential use of FedBN (keeping batch normalization local).

#### 2.4 Centralized Baseline

- The same LSTM architecture trained on pooled data from both furnaces (using only the shared feature set).
- Same hyperparameter search space as the FL setup for fair comparison.
- Evaluation on the same held-out test splits per furnace.

#### 2.5 Evaluation Metrics

- **Regression metrics:** RMSE, MAE, R² for active power prediction.
- **FL-specific metrics:** Communication cost (total bytes transmitted), convergence speed (rounds to target loss), per-furnace performance gap (fairness).
- **Statistical robustness:** Multiple random seeds, report mean ± std.

---

## LaTeX Conventions

Follow these formatting rules strictly:

- **Do not create new `.tex` files.** Write content into the existing template's section structure. If the template uses `\input{}` for sections, write into the corresponding files. If it's a single-file template, insert content at the appropriate `\section{}` markers.
- **Before writing anything**, read the full template to understand its structure, document class, packages loaded, and any custom commands or environments defined.
- Use `\cite{}` for all references — never write inline author-year manually.
- Use `\label{}` and `\ref{}` for all cross-references (sections, figures, tables, equations).
- Label convention: `sec:introduction`, `sec:methodology`, `sec:data`, `sec:lstm-architecture`, `sec:fl-setup`, `sec:baseline`, `sec:evaluation`, `tab:features`, `fig:fl-architecture`, `eq:fedavg`.
- Tables: use `booktabs` (`\toprule`, `\midrule`, `\bottomrule`). No vertical rules.
- Equations: use `equation` or `align` environments. Number all equations. Define FedAvg aggregation formula:
  ```latex
  w^{t+1} = \sum_{k=1}^{K} \frac{n_k}{n} w_k^{t+1}
  ```
- Use `\textbf{}` sparingly. Never use `\textbf` for emphasis in running text — use `\emph{}` if needed.
- Keep paragraphs focused. One idea per paragraph.
- British or American English — be consistent. Default to American English unless the template specifies otherwise.

---

## Anonymisation Rules

This is critical:

- **Never** include real facility names, company names, geographic identifiers, or specific plant/unit designators in the paper text.
- Refer to furnaces as **Furnace 1** and **Furnace 2** (not Furnace 42 / Furnace 44).
- Do not include raw tag names (e.g., `P42.AB0052`) in the paper. Describe features generically (e.g., "active power," "electrode position for phase A").
- The tag mapping file (`furnace_tag_mapping.md`) is an internal reference only — its contents should inform your understanding of the data but not appear verbatim in the paper.

---

## Workflow Instructions for Claude Code

1. **First**, read the entire LaTeX template directory. Understand the document class, loaded packages, existing sections, and any `\input{}` structure.
2. **Second**, read the `.bib` file to identify available citation keys. Map the references listed above to their actual BibTeX keys.
3. **Third**, read `furnace_tag_mapping.md` for data context (but remember: anonymise everything).
4. **Fourth**, read the literature review (`Federated_Learning_for_LSTM_Models_on_Equipment_Sensor_Data__A_Literature_Review.md`) for detailed claims and citations to use.
5. **Then** write the Introduction and Methodology sections following the structure above.
6. After writing, **compile the LaTeX document** (`pdflatex` + `bibtex` + `pdflatex` × 2) and fix any compilation errors.
7. Flag all `% TODO` and `% PROVISIONAL` comments in your output so they can be reviewed.

---

## What NOT to Do

- Do not fabricate experimental results or specific numbers that have not been produced yet.
- Do not invent references. Only cite what exists in the `.bib` file.
- Do not rewrite or modify the Literature Review section — it is complete.
- Do not modify the `.bib` file.
- Do not change the document class, page layout, or global formatting of the template.
- Do not include code snippets or implementation details in the Methodology — this is a research paper, not documentation.

