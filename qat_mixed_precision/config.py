"""Single source of truth for the online mixed-precision QAT study.

Pre-registered: bit set, budget schedule, controller thresholds, the run list.
Any post-hoc change must be labelled as such in README.md.
"""
from __future__ import annotations
import os
from pathlib import Path

# ---------------------------------------------------------------------------
# Models  (quantize only the projection Linears named here; everything else FP)
# ---------------------------------------------------------------------------
MODELS = {
    "mamba-130m": {
        "hf_id": "state-spaces/mamba-130m-hf",
        "kind": "mamba",
        # HF Mamba mixer projections (nn.Linear). conv1d / scan / norms excluded.
        "target_suffixes": ["in_proj", "x_proj", "dt_proj", "out_proj"],
    },
    "mamba2-130m": {
        "hf_id": "state-spaces/mamba2-130m",
        "kind": "mamba2",
        "target_suffixes": ["in_proj", "out_proj"],
    },
    "gpt2": {
        "hf_id": "gpt2",
        "kind": "gpt2",
        # GPT-2 uses transformers Conv1D (not nn.Linear) — wrapper handles layout.
        "target_suffixes": ["c_attn", "c_proj", "c_fc"],
    },
}

PRIMARY_MODEL = "mamba-130m"

# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------
TRAIN_DATASET = ("wikitext", "wikitext-103-raw-v1")
EVAL_DATASET  = ("wikitext", "wikitext-2-raw-v1")   # KL-Lens headline metric
SEQ_LEN       = int(os.environ.get("QMP_SEQ_LEN", 1024))
CALIB_SAMPLES = 16     # calib batch for marginal-KL audits
CALIB_SEQ_LEN = 512    # shorter context for cheap audits

# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------
TRAIN_BATCH   = int(os.environ.get("QMP_TRAIN_BATCH", 8))
EVAL_BATCH    = 8
GRAD_ACCUM    = 1
TOTAL_STEPS   = int(os.environ.get("QMP_TOTAL_STEPS", 4000))
LEARNING_RATE = 5e-5
WEIGHT_DECAY  = 0.01
WARMUP_RATIO  = 0.05
MAX_GRAD_NORM = 1.0
POSTHOC_RECOVER_STEPS = 1000   # short recovery finetune after post-hoc assignment

# ---------------------------------------------------------------------------
# Quantization: bit set (weights). Activations fixed dynamic int8.
# ---------------------------------------------------------------------------
BIT_SET   = [2, 4, 8]
MAX_BITS  = 8
MIN_BITS  = 2
ACT_BITS  = 8   # dynamic per-token activation precision (fixed)

# ---------------------------------------------------------------------------
# Budget schedule (online method)  -- average param-weighted weight-bits
# ---------------------------------------------------------------------------
WARMUP_FRAC = 0.15   # [0, warmup): all 8-bit, gather stats
HOLD_FRAC   = 0.70   # [warmup, hold): anneal 8 -> B*;  [hold, 1]: hold at B*, co-adapt
AUDIT_EVERY = 250    # steps between marginal-KL audits
MAX_AUDIT_LAYERS = 64  # subsample candidates per audit for speed (None = all)

# ---------------------------------------------------------------------------
# Controller thresholds (pre-registered; data-relative where noted)
# ---------------------------------------------------------------------------
RECOVER_PCTL  = 90    # raise a layer if marginal_up > p<RECOVER_PCTL> of audit dist
COOLDOWN_STEPS = 2 * AUDIT_EVERY   # lock a layer after a bit change
RAISE_CAP     = 3     # max precision-raises per layer (guarantees termination)

# ---------------------------------------------------------------------------
# The run list (Section 6 of TASK.md)
# ---------------------------------------------------------------------------
PRIMARY_SEEDS   = [0, 1, 2]
SECONDARY_SEEDS = [0]

UNIFORM_BITS    = [2, 4, 8]
BUDGET_TARGETS  = [3, 4, 5, 6]            # average-bit targets B*
SECONDARY_BUDGETS = [4, 6]
SECONDARY_UNIFORM = [4, 8]


def build_runs():
    """Enumerate every run as a dict. Crash-safe id = model__method__tag__seedN."""
    runs = []

    def add(model, method, tag, seed, **extra):
        runs.append(dict(
            run_id=f"{model}__{method}__{tag}__seed{seed}",
            model=model, method=method, tag=tag, seed=seed, **extra))

    # ---- primary: mamba-130m, 3 seeds ----
    for seed in PRIMARY_SEEDS:
        for b in UNIFORM_BITS:
            add(PRIMARY_MODEL, "uniform", f"b{b}", seed, bits=b)
        add(PRIMARY_MODEL, "posthoc_base", "base8", seed, bits=8)
        for B in BUDGET_TARGETS:
            add(PRIMARY_MODEL, "posthoc", f"B{B}", seed, budget=B)
            add(PRIMARY_MODEL, "online",  f"B{B}", seed, budget=B)

    # ---- second model + transfer, 1 seed ----
    for model in ["mamba2-130m", "gpt2"]:
        for seed in SECONDARY_SEEDS:
            for b in SECONDARY_UNIFORM:
                add(model, "uniform", f"b{b}", seed, bits=b)
            add(model, "posthoc_base", "base8", seed, bits=8)
            for B in SECONDARY_BUDGETS:
                add(model, "posthoc", f"B{B}", seed, budget=B)
                add(model, "online",  f"B{B}", seed, budget=B)
    return runs


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT    = Path(__file__).parent
RESULTS = ROOT / "results"
RUNS    = RESULTS / "runs"
AUDITS  = RESULTS / "audits"
FIGS    = RESULTS / "figures"
CKPTS   = RESULTS / "ckpts"        # shared posthoc base checkpoints
# NB: PARAM Shavak forbids writing to any path containing "log"; use "console".
CONSOLE = Path(os.environ.get("QMP_CONSOLE_DIR", str(ROOT / "console")))

for _p in (RESULTS, RUNS, AUDITS, FIGS, CKPTS, CONSOLE):
    _p.mkdir(parents=True, exist_ok=True)
