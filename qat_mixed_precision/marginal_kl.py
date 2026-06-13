"""Forward-only marginal-KL audit (KL-Lens direction baked in).

Fixes the prior project's bug: we measure with the ACTUAL quantized numerics at
each layer's real precision, referenced to the FP teacher. Nothing is silently
re-calibrated to a best-case scale.

KL direction (student->teacher, the only correct one per KL-Lens Prop. 1 / asymmetry):
    P = softmax(teacher_logits)   (all target layers FP)
    Q = softmax(student_logits)   (target layers at current/probe bits)
    KL_s->t = Σ Q·(log Q − log P)

For each layer we report:
    marginal_down(ℓ) = KL(b_ℓ−1) − KL(current)   # cost of dropping ℓ one level (≥0)
    marginal_up(ℓ)   = KL(current) − KL(b_ℓ+1)    # distortion removed by raising ℓ (≥0)
"""
from __future__ import annotations
import random
from typing import Dict, List, Optional
import torch
import torch.nn.functional as F

from config import MAX_BITS, MIN_BITS, MAX_AUDIT_LAYERS
from dyn_precision_linear import get_dp_layers


def _logits(model, batch) -> torch.Tensor:
    return model(input_ids=batch["input_ids"],
                 attention_mask=batch.get("attention_mask")).logits


def _kl_st(student_logits: torch.Tensor, logP: torch.Tensor) -> float:
    """KL(softmax(student) ‖ teacher), mean over tokens. logP = log_softmax(teacher)."""
    logQ = F.log_softmax(student_logits, dim=-1)
    Q = logQ.exp()
    kl = (Q * (logQ - logP)).sum(dim=-1).mean().item()
    return max(kl, 0.0)


@torch.no_grad()
def audit(model, calib_batch, candidates: Optional[List[str]] = None,
          want_down: bool = True, want_up: bool = True,
          subsample: Optional[int] = MAX_AUDIT_LAYERS) -> Dict[str, dict]:
    """Return {layer: {bits, marginal_down, marginal_up}} for audited layers."""
    model.eval()
    layers = get_dp_layers(model)
    names = list(layers.keys()) if candidates is None else \
        [n for n in candidates if n in layers]
    if subsample and len(names) > subsample:
        names = random.sample(names, subsample)

    # current per-layer bits (restore later)
    saved = {n: layers[n].bit_width for n in layers}

    # teacher: all target layers FP
    for m in layers.values():
        m.set_bits(None)
    teacher_logits = _logits(model, calib_batch)
    logP = F.log_softmax(teacher_logits, dim=-1)

    # current-config base KL (restore saved bits)
    for n, m in layers.items():
        m.set_bits(saved[n])
    base_kl = _kl_st(_logits(model, calib_batch), logP)

    out: Dict[str, dict] = {}
    for n in names:
        m = layers[n]
        b = saved[n] if saved[n] is not None else MAX_BITS
        rec = {"bits": b, "marginal_down": None, "marginal_up": None}
        if want_down and b > MIN_BITS:
            with m.override_bits(b - 1):
                kl = _kl_st(_logits(model, calib_batch), logP)
            rec["marginal_down"] = max(kl - base_kl, 0.0)
        if want_up and b < MAX_BITS:
            with m.override_bits(b + 1):
                kl = _kl_st(_logits(model, calib_batch), logP)
            rec["marginal_up"] = max(base_kl - kl, 0.0)
        out[n] = rec

    # restore exactly
    for n, m in layers.items():
        m.set_bits(saved[n])
    model.train()
    out["__base_kl__"] = {"base_kl": base_kl}  # sentinel for logging
    return out
