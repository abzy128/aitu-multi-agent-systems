"""Central configuration: paths, hyperparameters, device selection."""

from pathlib import Path

import torch

# ── Paths ──────────────────────────────────────────────────────────────────
ROOT = Path(__file__).parent.parent
DATA_DIR = ROOT / "data"
PROCESSED_DIR = DATA_DIR / "processed"
CHECKPOINTS_DIR = ROOT / "checkpoints"
ARTIFACTS_DIR = ROOT / "artifacts"
REPORTS_DIR = ROOT / "reports"

TRAIN_CSV = DATA_DIR / "train.csv"
TEST_CSV = DATA_DIR / "test.csv"

AUTOENCODER_CHECKPOINT = CHECKPOINTS_DIR / "autoencoder_best.pt"
CLASSIFIER_CHECKPOINT = CHECKPOINTS_DIR / "classifier_best.pt"
EVAL_RESULTS_JSON = ARTIFACTS_DIR / "eval_results.json"

# Create dirs on import
for _d in (PROCESSED_DIR, CHECKPOINTS_DIR, ARTIFACTS_DIR, REPORTS_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# ── Device ─────────────────────────────────────────────────────────────────
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ── Preprocessing ──────────────────────────────────────────────────────────
VAL_SPLIT = 0.1          # fraction of training data held out for validation
RANDOM_SEED = 42

# ── Shared training ────────────────────────────────────────────────────────
BATCH_SIZE = 256
NUM_WORKERS = 4

# ── Autoencoder ────────────────────────────────────────────────────────────
AE_LATENT_DIM = 32
AE_HIDDEN_DIMS = [64, 48]   # encoder layers (decoder mirrors these)
AE_LR = 1e-3
AE_EPOCHS = 50
AE_PATIENCE = 7             # early stopping patience
AE_THRESHOLD_PERCENTILE = 95  # percentile of normal recon error used as decision threshold

# ── Classifier ─────────────────────────────────────────────────────────────
CLS_HIDDEN_DIMS = [128, 64, 32]
CLS_LR = 1e-3
CLS_EPOCHS = 50
CLS_PATIENCE = 7

# ── Labels ─────────────────────────────────────────────────────────────────
ATTACK_CATEGORIES = ["DoS", "Probe", "R2L", "U2R"]
# NSL-KDD label → attack category mapping
LABEL_TO_CATEGORY = {
    "normal": "normal",
    # DoS
    "back": "DoS", "land": "DoS", "neptune": "DoS", "pod": "DoS",
    "smurf": "DoS", "teardrop": "DoS", "apache2": "DoS", "udpstorm": "DoS",
    "processtable": "DoS", "worm": "DoS",
    # Probe
    "ipsweep": "Probe", "nmap": "Probe", "portsweep": "Probe",
    "satan": "Probe", "mscan": "Probe", "saint": "Probe",
    # R2L
    "ftp_write": "R2L", "guess_passwd": "R2L", "imap": "R2L",
    "multihop": "R2L", "phf": "R2L", "spy": "R2L", "warezclient": "R2L",
    "warezmaster": "R2L", "sendmail": "R2L", "named": "R2L",
    "snmpgetattack": "R2L", "snmpguess": "R2L", "xlock": "R2L",
    "xsnoop": "R2L", "httptunnel": "R2L",
    # U2R
    "buffer_overflow": "U2R", "loadmodule": "U2R", "perl": "U2R",
    "rootkit": "U2R", "ps": "U2R", "sqlattack": "U2R",
    "xterm": "U2R",
}
CLASS_NAMES = ["normal"] + ATTACK_CATEGORIES   # 5 classes for multi-class classifier
NUM_CLASSES = len(CLASS_NAMES)
