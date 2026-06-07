"""Forward-only per-layer KL sensitivity audit (Kong et al. 2026, §4.8).

KL direction: student→teacher = KL(Q ‖ P), Q=softmax(quantized), P=softmax(fp)
  = Σ Q·(log Q − log P)

This direction correlates with perplexity (τ≈0.79); teacher→student is negatively
correlated and must NOT be used (Prop. 1 of KL Lens paper).

Cost: O(L) forward passes, forward-only, no backprop.
Audits only the layers passed in (frozen + borderline during training).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional
from data import move_batch
from qat_linear import DynStatQATLinear


def compute_layer_kl(
    model: nn.Module,
    calib_batch: dict,
    layer_names: Optional[List[str]] = None,
    device: torch.device = torch.device("cpu"),
) -> Dict[str, float]:
    """
    Compute KL_s->t for each specified layer (or all QAT layers).

    For each layer l:
      1. Set l to frozen with its current scale (simulating quantization of l only).
      2. Run forward → get logits (student).
      3. Unfreeze l, run forward → get logits (teacher/fp).
      4. KL = Σ Q·(log Q − log P)  where Q=softmax(student), P=softmax(fp).

    Returns dict: layer_name → KL (nats).
    """
    model.eval()
    batch = move_batch(calib_batch, device)
    # input_ids, attention_mask (drop labels for forward)
    fwd_keys = {k: v for k, v in batch.items() if k != "labels"}

    # collect all QAT layers if none specified
    all_layers = {name: m for name, m in model.named_modules() if isinstance(m, DynStatQATLinear)}
    if layer_names is None:
        layer_names = list(all_layers.keys())

    # baseline: all dynamic (teacher)
    _set_all_dynamic(all_layers)
    with torch.no_grad():
        teacher_logits = model(**fwd_keys).logits   # [B, C]
    P = F.softmax(teacher_logits, dim=-1)

    kl_scores: Dict[str, float] = {}

    for name in layer_names:
        if name not in all_layers:
            continue
        mod = all_layers[name]
        # capture scale from last forward (should already be set if training)
        if mod._last_mean_scale is None:
            # do a warm-up forward to capture scale
            _set_all_dynamic(all_layers)
            with torch.no_grad():
                model(**fwd_keys)
        scale = mod._last_mean_scale or 1.0
        zp    = mod._last_mean_zp or 0.0

        # freeze only this layer
        mod.freeze(scale=scale, zero_point=zp)
        with torch.no_grad():
            student_logits = model(**fwd_keys).logits
        mod.unfreeze()

        Q = F.softmax(student_logits, dim=-1)
        # KL(Q ‖ P) = Σ Q·(log Q − log P)  (student→teacher)
        kl = (Q * (Q.log() - P.log())).sum(dim=-1).mean().item()
        kl_scores[name] = max(kl, 0.0)   # numerical floor at 0

    # restore all dynamic
    _set_all_dynamic(all_layers)
    model.train()
    return kl_scores


def _set_all_dynamic(layers: dict):
    for m in layers.values():
        if m.frozen:
            m.unfreeze()
