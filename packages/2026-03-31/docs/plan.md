# Case Study Plan: Anomaly Detection for Network Security

## 1. Overview

**Objective:** Build a PyTorch-based anomaly detection system that identifies malicious network traffic using autoencoder and classification approaches. Train, evaluate, and produce a reproducible report.

**Dataset:** [NSL-KDD](https://www.unb.ca/cic/datasets/nsl.html) — an improved version of the original KDD Cup 1999 dataset. It removes redundant records and is the de-facto benchmark for network intrusion detection research.

- ~125K training samples, ~22K test samples
- 41 features (TCP connection attributes, content features, traffic features)
- Labels: `normal` + 4 attack categories (`DoS`, `Probe`, `R2L`, `U2R`)

**Stack:** Python 3.12 · PyTorch (CUDA) · uv · Jupyter · md2pdf

---

## 2. File Structure

```
anomaly-detection-network-security/
│
├── .gitignore
├── .python-version                  # pinned python version for uv
├── pyproject.toml                   # project metadata, dependencies
├── uv.lock                         # lockfile (auto-generated)
├── README.md
├── PLAN.md                          # this file
│
├── data/
│   └── download_dataset.py          # script to fetch & cache NSL-KDD
│
├── src/
│   ├── __init__.py
│   ├── config.py                    # hyperparams, paths, device selection
│   ├── dataset.py                   # PyTorch Dataset / DataLoader factories
│   ├── preprocessing.py             # encoding, scaling, feature engineering
│   ├── models.py                    # model architectures (Autoencoder, Classifier)
│   ├── train_utils.py               # training loop, early stopping, checkpointing
│   ├── eval_utils.py                # metrics, confusion matrix, ROC, threshold tuning
│   └── visualize.py                 # plotting helpers (loss curves, ROC, t-SNE, etc.)
│
├── notebooks/
│   ├── 01_eda.ipynb                 # exploratory data analysis
│   ├── 02_preprocessing.ipynb       # run & verify preprocessing pipeline
│   ├── 03_train_autoencoder.ipynb   # train autoencoder for unsupervised anomaly detection
│   ├── 04_train_classifier.ipynb    # train supervised classifier for comparison
│   ├── 05_evaluation.ipynb          # full evaluation, plots, comparison table
│   └── 06_generate_report.ipynb     # assemble markdown report → convert to PDF
│
├── reports/
│   ├── report.md                    # generated evaluation report (markdown)
│   └── report.pdf                   # converted from report.md via md2pdf
│
├── checkpoints/                     # saved model weights (.pt)
└── artifacts/                       # exported plots, figures (.png)
```

---

## 3. Environment & Dependency Setup

Managed entirely with **uv**.

### pyproject.toml — key dependencies

| Category       | Packages                                                        |
| -------------- | --------------------------------------------------------------- |
| Core ML        | `torch`, `torchvision` (CUDA build)                             |
| Data           | `pandas`, `numpy`, `scikit-learn`                                |
| Visualization  | `matplotlib`, `seaborn`                                          |
| Notebooks      | `jupyterlab`, `ipykernel`                                        |
| Report         | `md2pdf`, `jinja2` (for templating report.md)                    |
| Utilities      | `tqdm`, `pyyaml`                                                 |

### Bootstrap commands

```bash
uv init anomaly-detection-network-security
cd anomaly-detection-network-security
uv python install 3.12
uv add torch torchvision --extra-index-url https://download.pytorch.org/whl/cu124
uv add pandas numpy scikit-learn matplotlib seaborn
uv add jupyterlab ipykernel tqdm pyyaml jinja2 md2pdf
```

---

## 4. Dataset Acquisition

**File:** `data/download_dataset.py`

**Source URLs (public, direct download):**

```
https://raw.githubusercontent.com/defcom17/NSL_KDD/master/KDDTrain+.txt
https://raw.githubusercontent.com/defcom17/NSL_KDD/master/KDDTest+.txt
```

**Script responsibilities:**

1. Download `KDDTrain+.txt` and `KDDTest+.txt` if not already cached in `data/`.
2. Assign column headers (the raw files have no header row — use the 41 standard KDD feature names + `label` + `difficulty`).
3. Save processed files as `data/train.csv` and `data/test.csv`.
4. Print summary stats (row counts, class distribution) to verify integrity.

Can be invoked standalone:

```bash
uv run python data/download_dataset.py
```

---

## 5. Source Modules (`src/`)

Each module is imported by notebooks — no standalone training scripts needed.

| Module              | Responsibility                                                                                                        |
| ------------------- | --------------------------------------------------------------------------------------------------------------------- |
| `config.py`         | Central config: paths, hyperparams (lr, batch size, epochs, latent dim), device auto-selection (`cuda` → `cpu` fallback). |
| `preprocessing.py`  | Label encoding of categoricals (`protocol_type`, `service`, `flag`). Binary label (normal/attack) + multi-class label. `StandardScaler` fit/transform. Train/val split. |
| `dataset.py`        | `NetworkTrafficDataset(torch.utils.data.Dataset)` wrapping numpy arrays. Factory function returning `DataLoader` with configurable batch size and workers. |
| `models.py`         | **Autoencoder**: symmetric encoder-decoder, configurable latent dim, ReLU activations, trained with MSE reconstruction loss. **Classifier**: feedforward network, softmax over 5 classes (normal + 4 attack types). Both models `.to(device)`. |
| `train_utils.py`    | Generic `train_one_epoch()`, `validate()`. `EarlyStopping` class. Model checkpoint save/load. Training history dict. |
| `eval_utils.py`     | Compute accuracy, precision, recall, F1 (binary & per-class). Confusion matrix. ROC-AUC. Reconstruction error threshold selection (percentile-based) for autoencoder. |
| `visualize.py`      | `plot_loss_curves()`, `plot_confusion_matrix()`, `plot_roc_curve()`, `plot_reconstruction_error_dist()`, `plot_tsne()` — all save to `artifacts/`. |

---

## 6. Notebooks — Execution Order

All notebooks use `uv run jupyter lab` and import from `src/`.

### 01_eda.ipynb

- Load `data/train.csv`.
- Class distribution bar chart.
- Feature correlation heatmap.
- Statistical summary of numerical features.
- Identify skewed / zero-variance features.

### 02_preprocessing.ipynb

- Run `preprocessing.py` pipeline end-to-end.
- Verify encoded shapes, scaler parameters.
- Save preprocessed tensors or numpy arrays to `data/processed/` (optional cache).

### 03_train_autoencoder.ipynb

- Instantiate `Autoencoder` from `models.py`.
- Train on **normal-only** traffic (unsupervised paradigm).
- Log training & validation reconstruction loss per epoch.
- Save best checkpoint to `checkpoints/autoencoder_best.pt`.
- Plot loss curves.

### 04_train_classifier.ipynb

- Instantiate `Classifier` from `models.py`.
- Train on full labeled dataset (supervised paradigm).
- Log train/val loss + accuracy per epoch.
- Save best checkpoint to `checkpoints/classifier_best.pt`.
- Plot loss curves.

### 05_evaluation.ipynb

- Load both saved models.
- **Autoencoder evaluation:**
  - Compute reconstruction errors on test set.
  - Select threshold (e.g., 95th percentile of normal reconstruction error).
  - Binary classification metrics (normal vs. attack).
  - Plot reconstruction error distribution (normal vs. attack overlay).
- **Classifier evaluation:**
  - Per-class precision, recall, F1.
  - Confusion matrix heatmap.
  - ROC curves (one-vs-rest).
- **Comparison table:** Autoencoder (unsupervised) vs. Classifier (supervised) — accuracy, F1, AUC.
- t-SNE visualization of latent space (autoencoder bottleneck).
- Save all figures to `artifacts/`.
- Export evaluation results dict as `artifacts/eval_results.json`.

### 06_generate_report.ipynb

- Load `artifacts/eval_results.json`.
- Render `reports/report.md` using Jinja2 template with evaluation metrics and figure paths.
- Convert to PDF:
  ```python
  from md2pdf.core import md2pdf
  md2pdf("reports/report.pdf", md_file_path="reports/report.md")
  ```
- Verify PDF output exists.

---

## 7. Report Structure (`reports/report.md`)

The generated markdown report will contain:

1. **Abstract** — one-paragraph summary of the study.
2. **Dataset Description** — source, features, class distribution table.
3. **Methodology** — preprocessing steps, model architectures, training setup (optimizer, lr, epochs, device).
4. **Results**
   - Training curves (embedded images from `artifacts/`).
   - Autoencoder: reconstruction error distribution plot, binary metrics table.
   - Classifier: confusion matrix, per-class metrics table, ROC curves.
   - Comparison table (Autoencoder vs. Classifier).
   - t-SNE latent space visualization.
5. **Discussion** — strengths, limitations, observations.
6. **Conclusion** — key takeaways, potential next steps.

---

## 8. `.gitignore`

```gitignore
# Python
__pycache__/
*.py[cod]
*.pyo
*.egg-info/
dist/
build/

# Virtual environment (uv)
.venv/

# Jupyter
.ipynb_checkpoints/
*_executed.ipynb

# Artifacts & outputs
artifacts/*.png
reports/report.pdf
checkpoints/*.pt

# Data (downloaded, not committed)
data/*.txt
data/*.csv
data/processed/

# OS
.DS_Store
Thumbs.db

# IDE
.vscode/
.idea/

# md2pdf / wkhtmltopdf cache
*.wkhtmltopdf
```

---

## 9. Execution Checklist

```
[ ] uv init & install all dependencies
[ ] uv run python data/download_dataset.py
[ ] uv run jupyter lab
[ ] Run 01_eda.ipynb
[ ] Run 02_preprocessing.ipynb
[ ] Run 03_train_autoencoder.ipynb       (CUDA)
[ ] Run 04_train_classifier.ipynb        (CUDA)
[ ] Run 05_evaluation.ipynb
[ ] Run 06_generate_report.ipynb
[ ] Verify reports/report.pdf exists
[ ] git add & commit
```
