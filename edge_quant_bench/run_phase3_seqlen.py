"""
Phase 3: seq-len sensitivity sweep {32, 128, 512} on bert-base + gpt2.
Runs only the compile backend with key quant schemes.
"""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))

import torch
import torch._dynamo
torch._dynamo.config.suppress_errors = True

try:
    import transformers.utils.output_capturing as _oc
    _oc.__dict__["torch"] = torch
    _oc.is_torchdynamo_compiling = lambda: torch.compiler.is_compiling()
except Exception:
    pass

import config as C
from run_sweep import run_phase1

if __name__ == "__main__":
    # Seq-len sweep: bert-base + gpt2, key schemes, compile only, seq {32, 128, 512}
    for seq in C.SEQ_LENS_PHASE3:
        if seq == C.SEQ_LEN_DEFAULT:
            continue  # already done in Phase 1
        print(f"\n=== Phase 3: seq_len={seq} ===")
        run_phase1(
            models=["bert-base", "gpt2"],
            schemes=["fp16", "w8a16", "w4a16", "w8a8"],
            backends=["compile"],
            batches=[1, 16, 64],
            seq=seq,
        )
