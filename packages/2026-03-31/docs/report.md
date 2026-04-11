# Anomaly Detection for Network Security
## Case Study Report — NSL-KDD Dataset

**Course:** AITU Multi-Agent Systems  
**Dataset:** NSL-KDD (Network Intrusion Detection)  
**Hardware:** NVIDIA GeForce RTX 4060 Laptop GPU (CUDA)  
**Stack:** Python 3.12 · PyTorch · scikit-learn · pandas · seaborn

---

## 1. Abstract

This case study implements and evaluates two complementary anomaly detection approaches on the NSL-KDD network intrusion detection benchmark. An **autoencoder** (unsupervised) is trained exclusively on normal traffic and uses reconstruction error as an anomaly score; a **feedforward classifier** (supervised) is trained end-to-end over five traffic categories. Both models are implemented in PyTorch, trained on an RTX 4060 GPU, and evaluated on the standard NSL-KDD test split. The autoencoder achieves F1 = 0.877 for binary (normal/attack) detection, while the supervised classifier reaches 99.5 % validation accuracy on the full multi-class problem but generalises imperfectly to rare attack classes in the test set.

---

## 2. Dataset Description

### Source

**NSL-KDD** — an improved version of the KDD Cup 1999 dataset that removes duplicate records and balances class representation. Hosted publicly by the Canadian Institute for Cybersecurity.

| Split | Samples |
|-------|--------:|
| Training | 125,973 |
| Test | 22,544 |

### Features

41 features per TCP connection record, grouped into:
- **Basic features** (9): duration, protocol type, service, flag, bytes sent/received, land, wrong fragment, urgent.
- **Content features** (13): hot, failed logins, logged in, compromised, root shell, su attempted, file creations, shells, access files, outbound commands, host/guest login.
- **Traffic features** (19): count, srv_count, error rates, same/diff service rates, destination host aggregates.

Three features are categorical (`protocol_type`, `service`, `flag`); the remaining 38 are numerical.

### Class Distribution (Training Set)

| Class | Category | Count | % |
|-------|----------|------:|--:|
| normal | — | 67,343 | 53.5% |
| neptune | DoS | 41,214 | 32.7% |
| satan | Probe | 3,633 | 2.9% |
| ipsweep | Probe | 3,599 | 2.9% |
| portsweep | Probe | 2,931 | 2.3% |
| smurf | DoS | 2,646 | 2.1% |
| nmap | Probe | 1,493 | 1.2% |
| back | DoS | 956 | 0.8% |
| warezclient | R2L | 890 | 0.7% |
| *other* | various | ~1,268 | 1.0% |

The training set is dominated by `normal` and `neptune` (DoS). Rare classes — particularly U2R (≤30 samples) — are severely underrepresented, which directly impacts generalisation.

---

## 3. Methodology

### 3.1 Preprocessing

1. **Categorical encoding:** `LabelEncoder` fitted jointly on train + test for `protocol_type`, `service`, and `flag` to avoid unseen-label errors at test time.
2. **Label mapping:** Each of 39 attack subtypes is mapped to one of four categories (DoS, Probe, R2L, U2R). A binary label (0 = normal, 1 = attack) is also generated.
3. **Scaling:** `StandardScaler` fitted on training features only, applied to validation and test splits.
4. **Train/val split:** 10% stratified hold-out (113,375 train / 12,598 val).

### 3.2 Autoencoder (Unsupervised)

**Architecture:**

```
Input (41) → Linear(64) → BN → ReLU
           → Linear(48) → BN → ReLU
           → Linear(32)          ← latent space
           → Linear(48) → BN → ReLU
           → Linear(64) → BN → ReLU
           → Output (41)
```

- **Training data:** Normal traffic only (60,608 samples) — the model never sees attack labels.
- **Loss:** Mean Squared Error (reconstruction loss).
- **Optimiser:** Adam, lr = 1e-3, batch size = 256.
- **Early stopping:** patience = 7 epochs.
- **Decision rule:** A test sample is flagged as an attack if its reconstruction error exceeds the **95th percentile** of normal-traffic training errors (threshold = 0.0724).

**Training outcome:** 18 epochs (early stopped). Best validation loss = 0.2669 at epoch 11.

### 3.3 Classifier (Supervised)

**Architecture:**

```
Input (41) → Linear(128) → BN → ReLU → Dropout(0.3)
           → Linear(64)  → BN → ReLU → Dropout(0.3)
           → Linear(32)  → BN → ReLU → Dropout(0.3)
           → Linear(5)                ← 5 classes
```

- **Training data:** Full labelled dataset (all 5 classes).
- **Loss:** Cross-Entropy.
- **Optimiser:** Adam, lr = 1e-3, batch size = 256.
- **Early stopping:** patience = 7 epochs.

**Training outcome:** 26 epochs (early stopped). Best validation accuracy = 99.57% at epoch 20.

---

## 4. Results

### 4.1 Training Curves

**Autoencoder reconstruction loss** converged smoothly over 18 epochs. Training loss fell from 0.444 → 0.096; validation loss plateaued around 0.267, indicating the model learns a compact normal-traffic manifold.

**Classifier** converged rapidly — validation accuracy exceeded 98.5% after only 1 epoch, reaching 99.57% at peak. Cross-entropy loss decreased from 0.242 → 0.023.

### 4.2 Autoencoder Evaluation (Binary: Normal vs. Attack)

Threshold selected at the 95th percentile of normal-traffic reconstruction errors on the training set:

**Threshold = 0.0724**

| Metric | Value |
|--------|------:|
| Accuracy | **87.12%** |
| Precision | **96.28%** |
| Recall | **80.49%** |
| F1 Score | **87.68%** |

**Interpretation:** High precision (96.3%) means the autoencoder rarely raises false alarms — when it flags traffic as anomalous, it is almost certainly an attack. Lower recall (80.5%) means some attacks (particularly those that produce low reconstruction error, e.g. subtle R2L/U2R patterns) are missed. The reconstruction error distribution shows a clear separation between normal and attack traffic, validating the threshold-based approach.

### 4.3 Classifier Evaluation (Multi-class)

#### Overall Metrics

| Metric | Value |
|--------|------:|
| Accuracy | **76.74%** |
| Macro F1 | **48.86%** |
| ROC-AUC (macro OvR) | **92.52%** |

#### Per-Class Breakdown

| Class | Precision | Recall | F1 | Support |
|-------|----------:|-------:|---:|--------:|
| normal | 67.51% | 97.65% | 79.83% | 9,711 |
| DoS | 95.43% | 82.49% | 88.49% | 7,460 |
| Probe | 81.15% | 67.20% | 73.52% | 2,421 |
| R2L | 81.82% | 1.25% | **2.46%** | 2,885 |
| U2R | 0.00% | 0.00% | **0.00%** | 67 |

**Interpretation:**
- **DoS** is detected with high confidence (F1 = 88.5%) — the high-volume flooding patterns in the training set generalise well.
- **Probe** achieves F1 = 73.5%, reflecting reasonable generalisation of port-scan signatures.
- **R2L** (Remote-to-Local) collapses almost entirely: recall = 1.25%, F1 = 2.5%. The test set contains many R2L subtypes (`guess_passwd`, `warezmaster`, `snmpguess`, etc.) not seen in training, causing the model to misclassify them — most are predicted as `normal` due to the class prior.
- **U2R** (User-to-Root) achieves F1 = 0%: only 67 test samples, zero correctly identified. This is a classic class-imbalance failure — the model learns to ignore U2R due to its near-zero frequency in training.
- The high ROC-AUC (92.5%) reflects that the model's probability estimates are well-ranked even when the argmax prediction is wrong, particularly for the majority classes.

#### Binary Performance of Classifier (Normal vs. Attack)

When the classifier's multi-class outputs are collapsed to binary:

| Metric | Value |
|--------|------:|
| Accuracy | 78.75% |
| Precision | 97.32% |
| Recall | 64.44% |
| F1 | 77.54% |

The supervised classifier has lower binary recall than the autoencoder (64.4% vs. 80.5%) because it misclassifies most R2L and all U2R attacks as normal.

### 4.4 Comparison: Autoencoder vs. Classifier (Binary)

| Model | Accuracy | Precision | Recall | F1 |
|-------|----------:|----------:|-------:|---:|
| Autoencoder (unsupervised) | 87.12% | 96.28% | 80.49% | 87.68% |
| Classifier (supervised) | 78.75% | 97.32% | 64.44% | 77.54% |

**Key finding:** The unsupervised autoencoder outperforms the supervised classifier in binary anomaly detection across all metrics except precision. This counter-intuitive result is explained by the R2L/U2R distributional shift: the test set contains many attack subtypes absent from training, which the classifier cannot recognise but the autoencoder detects as anomalous due to their unusual reconstruction signature.

### 4.5 t-SNE Latent Space Visualisation

The t-SNE projection of the autoencoder's 32-dimensional latent space (5,000-sample subsample) reveals that **normal traffic forms a tight, well-separated cluster**, while DoS and Probe attacks occupy distinct regions. R2L and U2R samples are more diffuse and partially overlap with normal traffic, explaining the lower recall for those classes.

---

## 5. Discussion

### Strengths

- **Autoencoder generalises to novel attacks.** Because it only models normality, any deviation — including unseen attack subtypes — raises the reconstruction error. This is critical for real-world deployment where attackers continuously evolve techniques.
- **High precision on both models.** Both approaches flag very few false positives (precision > 96%), which is operationally important to avoid alarm fatigue for security analysts.
- **Fast training on GPU.** Both models converge in under 30 seconds on the RTX 4060.

### Limitations

- **Class imbalance is a fundamental challenge.** U2R has only 30 training samples — no model can reliably detect it without oversampling (SMOTE), cost-sensitive loss, or few-shot learning.
- **Train/test distribution shift for R2L.** NSL-KDD's test set intentionally includes novel attack subtypes. The supervised classifier memorises training-set attack signatures rather than learning generalisable patterns.
- **Autoencoder threshold is static.** A fixed percentile threshold does not adapt to concept drift (e.g. gradual change in normal traffic patterns over time).
- **Reconstruction error is a scalar.** It does not distinguish *which* features deviate — root-cause analysis requires additional work.

### Observations

- Validation accuracy of 99.57% on the classifier versus 76.74% test accuracy reveals significant overfitting to the training-set attack distribution — the model is not learning robust attack features.
- The autoencoder's latent space naturally separates traffic types without any label supervision, suggesting that anomaly-detection autoencoders could serve as unsupervised feature extractors for downstream tasks.

---

## 6. Conclusion

Both paradigms are effective for network intrusion detection, with complementary strengths:

| Paradigm | Best for | Weakness |
|----------|----------|---------|
| Autoencoder (unsupervised) | Novel/zero-day attacks, no labels required | Misses low-error anomalies (some R2L/U2R) |
| Classifier (supervised) | High-volume known attacks (DoS, Probe) | Fails on unseen attack subtypes |

A **hybrid ensemble** — using the autoencoder's reconstruction error as an additional feature for the classifier, or routing low-confidence classifier predictions to the autoencoder — would likely improve overall detection. Future work should also address class imbalance for U2R/R2L via synthetic oversampling, weighted loss functions, or dedicated few-shot classifiers.

---

## Appendix: Training Configuration

| Parameter | Autoencoder | Classifier |
|-----------|-------------|-----------|
| Architecture | 41→64→48→32→48→64→41 | 41→128→64→32→5 |
| Activation | ReLU + BatchNorm | ReLU + BatchNorm + Dropout(0.3) |
| Loss | MSE | Cross-Entropy |
| Optimiser | Adam | Adam |
| Learning rate | 1e-3 | 1e-3 |
| Batch size | 256 | 256 |
| Max epochs | 50 | 50 |
| Early stopping | patience=7 | patience=7 |
| Epochs run | 18 | 26 |
| Device | CUDA (RTX 4060) | CUDA (RTX 4060) |
| Training samples | 60,608 (normal only) | 113,375 (all classes) |
