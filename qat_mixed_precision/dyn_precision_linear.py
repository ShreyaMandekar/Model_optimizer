"""DynPrecisionLinear: STE fake-quant linear with a settable weight bit-width.

- Weights: per-output-channel symmetric fake-quant at `bit_width` bits (STE).
  bit_width=None -> full-precision passthrough (the FP "teacher").
- Activations: dynamic per-token symmetric int8 fake-quant (fixed; ACT_BITS).
- Supports two weight layouts:
    'linear'  : weight [out, in]   (torch.nn.Linear; Mamba/Mamba2 projections)
    'conv1d'  : weight [in, out]   (transformers Conv1D; GPT-2)
- The weight Parameter is the SAME object after wrapping -> optimizer state intact.

This is a TRAINING-TIME accuracy study: fake-quant only, no real low-bit kernels.
"""
from __future__ import annotations
import contextlib
from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F

from config import ACT_BITS, MAX_BITS, MIN_BITS


def _round_ste(x: torch.Tensor) -> torch.Tensor:
    return x + (torch.round(x) - x).detach()


def fake_quant_weight(w: torch.Tensor, bits: int, ch_axis: int) -> torch.Tensor:
    """Per-output-channel symmetric weight fake-quant with STE."""
    if bits is None or bits >= 16:
        return w
    qmax = (1 << (bits - 1)) - 1            # b=2 -> 1 (ternary), 4 -> 7, 8 -> 127
    # amax over every axis except the output-channel axis
    reduce_dims = [d for d in range(w.dim()) if d != ch_axis]
    amax = w.detach().abs().amax(dim=reduce_dims, keepdim=True).clamp_min(1e-8)
    scale = amax / qmax
    wq = _round_ste(w / scale).clamp(-qmax, qmax) * scale
    return wq


def fake_quant_act_dynamic(x: torch.Tensor, bits: int = ACT_BITS) -> torch.Tensor:
    """Dynamic per-token symmetric activation fake-quant with STE."""
    qmax = (1 << (bits - 1)) - 1
    amax = x.detach().abs().amax(dim=-1, keepdim=True).clamp_min(1e-8)
    scale = amax / qmax
    return _round_ste(x / scale).clamp(-qmax, qmax) * scale


class DynPrecisionLinear(nn.Module):
    def __init__(self, weight: nn.Parameter, bias: Optional[nn.Parameter],
                 layout: str, in_features: int, out_features: int):
        super().__init__()
        assert layout in ("linear", "conv1d")
        self.layout = layout
        self.in_features = in_features
        self.out_features = out_features
        self.weight = weight            # SAME Parameter object (optimizer state safe)
        self.bias = bias
        # output-channel axis for per-channel weight quant
        self._ch_axis = 0 if layout == "linear" else 1
        self.bit_width: Optional[int] = MAX_BITS   # current weight precision
        self.quant_enabled = True                  # off -> pure FP (teacher)

    # -- precision control -------------------------------------------------
    def set_bits(self, b: Optional[int]):
        self.bit_width = b

    @contextlib.contextmanager
    def override_bits(self, b: Optional[int]):
        prev = self.bit_width
        self.bit_width = b
        try:
            yield
        finally:
            self.bit_width = prev

    def num_weight_params(self) -> int:
        return self.weight.numel()

    # -- forward -----------------------------------------------------------
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.quant_enabled or self.bit_width is None:
            w = self.weight
            xq = x
        else:
            w = fake_quant_weight(self.weight, self.bit_width, self._ch_axis)
            xq = fake_quant_act_dynamic(x, ACT_BITS)
        if self.layout == "linear":
            return F.linear(xq, w, self.bias)
        else:  # conv1d: y = x @ W + b , W is [in, out]
            out = torch.matmul(xq, w)
            if self.bias is not None:
                out = out + self.bias
            return out


# ---------------------------------------------------------------------------
# Wrapping
# ---------------------------------------------------------------------------

def _is_target(name: str, suffixes) -> bool:
    return any(name.split(".")[-1] == s for s in suffixes)


def _set_nested(model: nn.Module, name: str, new: nn.Module):
    parts = name.split(".")
    parent = model
    for p in parts[:-1]:
        parent = getattr(parent, p)
    setattr(parent, parts[-1], new)


def wrap_model(model: nn.Module, target_suffixes) -> tuple[nn.Module, list[str]]:
    """Replace every targeted Linear / Conv1D with DynPrecisionLinear.

    Returns (model, list_of_wrapped_names). Detects layout from module type.
    """
    try:
        from transformers.pytorch_utils import Conv1D
    except Exception:                       # pragma: no cover
        Conv1D = ()

    wrapped = []
    for name, module in list(model.named_modules()):
        if not _is_target(name, target_suffixes):
            continue
        if isinstance(module, nn.Linear):
            dp = DynPrecisionLinear(module.weight, module.bias, "linear",
                                    module.in_features, module.out_features)
        elif Conv1D and isinstance(module, Conv1D):
            # Conv1D.weight is [in, out]; nf = out_features
            in_f, out_f = module.weight.shape
            dp = DynPrecisionLinear(module.weight, module.bias, "conv1d", in_f, out_f)
        else:
            continue
        _set_nested(model, name, dp)
        wrapped.append(name)
    return model, wrapped


def get_dp_layers(model: nn.Module) -> dict[str, DynPrecisionLinear]:
    return {n: m for n, m in model.named_modules() if isinstance(m, DynPrecisionLinear)}


def set_all_bits(model: nn.Module, b: Optional[int]):
    for m in get_dp_layers(model).values():
        m.set_bits(b)


def avg_bits(model: nn.Module) -> float:
    """Param-weighted average weight-bits over the wrapped layers."""
    tot = wsum = 0.0
    for m in get_dp_layers(model).values():
        n = m.num_weight_params()
        b = m.bit_width if m.bit_width is not None else MAX_BITS
        wsum += n * b
        tot += n
    return wsum / max(tot, 1.0)


# ---------------------------------------------------------------------------
# Unit test
# ---------------------------------------------------------------------------

def run_unit_test():
    torch.manual_seed(0)
    lin = nn.Linear(64, 32)
    x = torch.randn(4, 10, 64)
    y_ref = lin(x)

    dp = DynPrecisionLinear(lin.weight, lin.bias, "linear", 64, 32)
    assert dp.weight is lin.weight, "weight object changed"
    dp.set_bits(None)
    assert torch.allclose(dp(x), y_ref, atol=1e-5), "FP passthrough != original"
    for b in (8, 4, 2):
        dp.set_bits(b)
        out = dp(x)
        assert out.shape == y_ref.shape
        err = (out - y_ref).abs().mean().item()
        print(f"[ok] linear b={b}: mean|Δ|={err:.4f}")

    # conv1d layout (GPT-2 style): weight [in, out]
    w = nn.Parameter(torch.randn(64, 32) * 0.1)
    b_ = nn.Parameter(torch.zeros(32))
    dpc = DynPrecisionLinear(w, b_, "conv1d", 64, 32)
    dpc.set_bits(None)
    y_fp = dpc(x)
    y_manual = torch.matmul(x, w) + b_
    assert torch.allclose(y_fp, y_manual, atol=1e-5), "conv1d FP mismatch"
    dpc.set_bits(4); _ = dpc(x)
    print("[ok] conv1d layout FP + 4-bit run")

    print("[PASS] DynPrecisionLinear unit test")
    return True


if __name__ == "__main__":
    run_unit_test()
