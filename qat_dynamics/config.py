"""Single source of truth for all experiment parameters.

Pre-registered thresholds (Section 4.4 of TASK.md). Any post-hoc changes must
be labelled as such in README.md.
"""
from dataclasses import dataclass, field
from typing import List, Tuple

# ---------------------------------------------------------------------------
# Models & tasks
# ---------------------------------------------------------------------------
MODELS = {
    "distilbert": {
        "hf_name": "distilbert-base-uncased",
        "kind": "encoder",
    },
    "bert": {
        "hf_name": "bert-base-uncased",
        "kind": "encoder",
    },
}

TASKS = {
    "sst2": {"glue_name": "sst2", "text_col": "sentence", "label_col": "label", "num_labels": 2},
    "mnli": {"glue_name": "mnli", "text_col1": "premise", "text_col2": "hypothesis", "label_col": "label", "num_labels": 3},
    "qnli": {"glue_name": "qnli", "text_col1": "question", "text_col2": "sentence", "label_col": "label", "num_labels": 2},
}

PRIMARY_MODEL = "distilbert"
PRIMARY_TASK  = "sst2"
SEEDS         = [42, 123, 7]

# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------
MAX_SEQ_LEN    = 128
TRAIN_BATCH    = 32
EVAL_BATCH     = 64
LEARNING_RATE  = 2e-5
WEIGHT_DECAY   = 0.01
NUM_EPOCHS     = 3
WARMUP_RATIO   = 0.1

# ---------------------------------------------------------------------------
# Quantization
# ---------------------------------------------------------------------------
# W4 weight / A8 dynamic activation via torchao Int8DynActInt4WeightQATQuantizer
QAT_SCHEME = "w4a8_dynamic"

# ---------------------------------------------------------------------------
# Controller & tracker  (pre-registered, Section 4.4)
# ---------------------------------------------------------------------------

# ---- Freeze triggers ----
CV_FREEZE_THRESHOLD  = 5.0      # CV% below which layer is freeze-eligible
CV_FREEZE_PATIENCE   = 50       # consecutive steps CV must stay below threshold
KL_FREEZE_THRESHOLD  = 0.01     # nats; KL_s->t below this → freeze eligible

# ---- Unfreeze triggers ----
CV_DRIFT_THRESHOLD   = 0.15     # |s_dyn - s_frozen| / s_frozen; drift-based unfreeze
CV_DRIFT_PATIENCE    = 3        # consecutive audits drift must hold
KL_UNFREEZE_THRESHOLD = 0.05    # nats; re-audit KL above this → unfreeze

# ---- Anti-thrashing ----
COOLDOWN_STEPS   = 150          # steps a layer stays in DYNAMIC after unfreeze
REFREEZE_CAP     = 2            # max times a layer can unfreeze; beyond → pinned DYNAMIC
AUDIT_EVERY      = 100          # steps between unfreeze audits (same for both columns)

# ---- Phase gate ----
FREEZE_PHASE_GATE = 0.1         # fraction of training elapsed before freeze-checks start

# ---- KL audit ----
KL_CALIB_SAMPLES = 64           # calibration batch size for KL sensitivity audits

# ---- EMA ----
EMA_ALPHA        = 0.15         # exponential moving average alpha for scale tracker

# ---- Regression safety net ----
REGRESSION_TOL   = 0.02         # val-loss regression fraction that triggers global audit

# ---- Soft-freeze (S4) ----
SOFT_EMA_MOMENTUM = 0.99        # high-momentum EMA; scale drifts very slowly

# ---------------------------------------------------------------------------
# Target frozen fraction (fairness control)
# ---------------------------------------------------------------------------
TARGET_FROZEN_FRACTION = 0.30   # 30% of QAT layers frozen at end of training

# ---------------------------------------------------------------------------
# Ablation grid  (metric × strategy)
# ---------------------------------------------------------------------------
METRICS     = ["cv", "kl"]
STRATEGIES  = ["s1_oneway", "s2_recal", "s3_reactive", "s4_soft"]

# All arms: list of (metric, strategy) tuples
ARMS: List[Tuple[str, str]] = [
    (m, s) for m in METRICS for s in STRATEGIES
]

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
LOG_FREQ_EARLY  = 10   # every N steps for first 20% of training
LOG_FREQ_LATE   = 50   # every N steps after 20%
EARLY_FRACTION  = 0.20 # boundary between early/late logging

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
from pathlib import Path
ROOT    = Path(__file__).parent
RESULTS = ROOT / "results"
RUNS    = RESULTS / "runs"
AUDITS  = RESULTS / "audits"
FIGS    = RESULTS / "figures"
LOGS    = ROOT / "logs"

for _p in [RESULTS, RUNS, AUDITS, FIGS, LOGS]:
    _p.mkdir(parents=True, exist_ok=True)
