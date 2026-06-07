"""DynStatQATLinear: unified QAT linear with freeze/unfreeze via a flag.

Design (TASK.md §4.1): wraps Int8DynActInt4WeightQATLinear.
  - frozen=False → dynamic activation scale (same numerics as stock torchao)
  - frozen=True  → activation scale pinned to frozen_scale/frozen_zero_point
Weight Parameter is the SAME object throughout; optimizer state is never disturbed.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchao.quantization.qat.linear import Int8DynActInt4WeightQATLinear
from torchao.quantization.qat.fake_quantizer import (
    _fake_quantize_per_token,
    _DTYPE_TO_QVALUE_BOUNDS,
    _choose_qparams_per_token_asymmetric,
)


class DynStatQATLinear(Int8DynActInt4WeightQATLinear):
    """Int8DynActInt4Weight QAT linear with bidirectional freeze/unfreeze.

    Supports an optional extra_bias (for models like DistilBERT/BERT whose
    nn.Linear layers have bias=True, which torchao's stock filter skips).
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.frozen = False
        # scalar buffers — captured EMA of per-token scale/zero_point
        self.register_buffer("frozen_scale", torch.tensor(1.0, dtype=torch.float32))
        self.register_buffer("frozen_zero_point", torch.tensor(0.0, dtype=torch.float32))
        self._last_mean_scale: float | None = None
        self._last_mean_zp: float | None = None
        self._extra_bias: nn.Parameter | None = None  # set by wrap_model for bias layers

    # ------------------------------------------------------------------
    # forward
    # ------------------------------------------------------------------

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.weight_fake_quantizer is not None:
            w = self.weight_fake_quantizer(self.weight)
        else:
            w = self.weight

        if self.activation_fake_quantizer is None or not self.activation_fake_quantizer.enabled:
            return F.linear(x, w, self.bias)

        if not self.frozen:
            x_fq = self.activation_fake_quantizer(x)
            # capture current mean scale for tracker / freeze capture
            if self.activation_fake_quantizer.scale is not None:
                self._last_mean_scale = self.activation_fake_quantizer.scale.detach().mean().item()
                self._last_mean_zp    = self.activation_fake_quantizer.zero_point.detach().float().mean().item()
        else:
            x_fq = self._frozen_fake_quant(x)

        out = F.linear(x_fq, w)
        if self._extra_bias is not None:
            out = out + self._extra_bias
        return out

    def _frozen_fake_quant(self, x: torch.Tensor) -> torch.Tensor:
        orig_shape = x.shape
        x2d = x.reshape(-1, x.shape[-1])
        n = x2d.shape[0]
        scale = self.frozen_scale.expand(n, 1)
        zp    = self.frozen_zero_point.expand(n, 1).to(torch.int32)
        qmin, qmax = _DTYPE_TO_QVALUE_BOUNDS[self.activation_fake_quantizer.config.dtype]
        x_fq = _fake_quantize_per_token(x2d, scale, zp, qmin, qmax)
        return x_fq.reshape(orig_shape)

    # ------------------------------------------------------------------
    # freeze / unfreeze API
    # ------------------------------------------------------------------

    def freeze(self, scale: float | None = None, zero_point: float = 0.0):
        """Pin activation scale. If scale is None, capture from last dynamic forward."""
        if scale is None:
            scale = self._last_mean_scale
        if scale is None:
            raise RuntimeError("Cannot freeze before a forward pass has captured a scale.")
        self.frozen_scale.fill_(scale)
        self.frozen_zero_point.fill_(zero_point)
        self.frozen = True

    def unfreeze(self):
        """Restore dynamic mode; seed internal scale from frozen value (no cold start)."""
        self.frozen = False
        if self.activation_fake_quantizer is not None:
            # seed so the first dynamic forward doesn't start from scratch
            self.activation_fake_quantizer.scale = self.frozen_scale.unsqueeze(0).clone()
            self.activation_fake_quantizer.zero_point = (
                self.frozen_zero_point.unsqueeze(0).to(torch.int32).clone()
            )
        self._last_mean_scale = self.frozen_scale.item()
        self._last_mean_zp    = self.frozen_zero_point.item()


# ---------------------------------------------------------------------------
# Factory: replace all QATLinear in a prepared model with DynStatQATLinear
# ---------------------------------------------------------------------------

def wrap_model(model: nn.Module, groupsize: int = 256) -> nn.Module:
    """Replace every nn.Linear (and Int8DynActInt4WeightQATLinear) with DynStatQATLinear.

    Handles bias layers (common in BERT/DistilBERT) by storing bias as _extra_bias.
    Skips layers whose in_features % groupsize != 0 (cannot be W4 quantized).
    """
    for name, module in list(model.named_modules()):
        if type(module) is Int8DynActInt4WeightQATLinear:
            dyn = _clone_as_dynstat(module)
            _set_nested_attr(model, name, dyn)
        elif type(module) is nn.Linear:
            if module.in_features % groupsize != 0:
                continue  # can't quantize this layer
            dyn = _from_linear(module, groupsize=groupsize)
            _set_nested_attr(model, name, dyn)
    return model


def _from_linear(src: nn.Linear, groupsize: int = 256) -> "DynStatQATLinear":
    """Create a DynStatQATLinear from a plain nn.Linear (possibly with bias)."""
    from torchao.quantization.qat import Int8DynActInt4WeightQATQuantizer
    # build a fresh QATLinear shell (no bias, weights copied)
    qat_shell = Int8DynActInt4WeightQATLinear(
        src.in_features, src.out_features, bias=False, groupsize=groupsize
    )
    qat_shell.weight = src.weight  # same tensor — preserves optimizer state
    dst = _clone_as_dynstat(qat_shell)
    # store bias separately
    if src.bias is not None:
        dst._extra_bias = nn.Parameter(src.bias.data.clone())
    return dst


def _clone_as_dynstat(src: Int8DynActInt4WeightQATLinear) -> "DynStatQATLinear":
    dst = DynStatQATLinear.__new__(DynStatQATLinear)
    dst.__dict__.update(src.__dict__)
    nn.Module.__init__(dst)
    for k, v in src._parameters.items():
        dst.register_parameter(k, v)  # same Parameter object — optimizer state preserved
    for k, v in src._buffers.items():
        dst.register_buffer(k, v)
    for k, child in src._modules.items():
        dst.add_module(k, child)
    dst.frozen = False
    dst.register_buffer("frozen_scale", torch.tensor(1.0, dtype=torch.float32))
    dst.register_buffer("frozen_zero_point", torch.tensor(0.0, dtype=torch.float32))
    dst._last_mean_scale = None
    dst._last_mean_zp    = None
    dst._extra_bias      = None
    return dst


def _set_nested_attr(model: nn.Module, name: str, new_module: nn.Module):
    parts = name.split(".")
    parent = model
    for p in parts[:-1]:
        parent = getattr(parent, p)
    setattr(parent, parts[-1], new_module)


# ---------------------------------------------------------------------------
# Unit test: DynStatQATLinear (dynamic mode) == stock Int8DynActInt4WeightQATLinear
# ---------------------------------------------------------------------------

def run_unit_test():
    import copy
    torch.manual_seed(0)
    in_f, out_f = 256, 256
    stock = Int8DynActInt4WeightQATLinear(in_f, out_f, bias=False)
    dyn   = _clone_as_dynstat(stock)

    # both must share the same weight tensor
    assert stock.weight is dyn.weight, "Weight objects differ after clone"

    stock.eval(); dyn.eval()
    x = torch.randn(4, 32, in_f)

    with torch.no_grad():
        y_stock = stock(x)
        y_dyn   = dyn(x)

    max_err = (y_stock - y_dyn).abs().max().item()
    assert max_err < 1e-4, f"Dynamic mode mismatch: max_err={max_err:.2e}"
    print(f"[PASS] DynStat dynamic == stock torchao  (max_err={max_err:.2e})")

    # freeze at captured scale and ensure forward still runs
    dyn.train()
    _ = dyn(x)   # capture scale
    scale_before = dyn._last_mean_scale
    dyn.freeze()
    y_frozen = dyn(x)
    print(f"[PASS] Frozen forward runs. Captured scale={scale_before:.6f}")

    # unfreeze and run again — optimizer-state pointer preserved
    param_id_before = id(dyn.weight)
    dyn.unfreeze()
    _ = dyn(x)
    param_id_after = id(dyn.weight)
    assert param_id_before == param_id_after, "Weight object changed after unfreeze!"
    print(f"[PASS] Unfreeze preserves weight object (optimizer state intact)")

    return True


if __name__ == "__main__":
    run_unit_test()
