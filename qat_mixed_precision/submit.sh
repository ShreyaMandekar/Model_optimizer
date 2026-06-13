#!/usr/bin/env bash
# PARAM Shavak entrypoint: env check -> Phase-0 sanity -> full sweep.
# Run from the qat_mixed_precision/ directory:  bash submit.sh
set -euo pipefail

PY="${QMP_PYTHON:-/home/shreya/venvs/fusion_qat/bin/python}"
cd "$(dirname "$0")"

echo "=== [0/4] environment ==="
$PY env_check.py

echo "=== [1/4] unit test: DynPrecisionLinear ==="
$PY dyn_precision_linear.py

echo "=== [2/4] wrap counts (sanity) ==="
$PY models.py mamba-130m

echo "=== [3/4] Phase-0 gate: FP baseline PPL (mamba-130m) ==="
# quick FP perplexity check before the sweep (no training)
$PY - <<'PYEOF'
import torch
from models import load_model_and_tokenizer
from data import make_eval_ids, eval_perplexity
from dyn_precision_linear import set_all_bits
m, tok, wrapped = load_model_and_tokenizer("mamba-130m")
dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
m.to(dev); set_all_bits(m, None)  # FP teacher
ids = make_eval_ids(tok)
print("FP WikiText-2 PPL (mamba-130m):", round(eval_perplexity(m, ids, dev), 3),
      "| wrapped layers:", len(wrapped))
print(">>> GATE: expect ~21-30. If much higher, STOP and reconsider testbed.")
PYEOF

echo "=== [4/4] full sweep ==="
$PY run_all.py "$@"

echo "=== analysis ==="
$PY analyze.py
$PY plots.py mamba-130m || true
echo "DONE."
