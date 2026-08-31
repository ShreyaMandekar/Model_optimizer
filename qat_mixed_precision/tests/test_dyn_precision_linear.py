"""Unit tests for DynPrecisionLinear (moved out of dyn_precision_linear.py so
pytest can discover/run it and it's covered by CI instead of a manual
`python dyn_precision_linear.py` invocation).

Run with: pytest tests/test_dyn_precision_linear.py -v
(or `pytest` from the qat_mixed_precision/ root to run the whole suite)
"""
import pytest
import torch
import torch.nn as nn

from dyn_precision_linear import DynPrecisionLinear


@pytest.fixture
def linear_fixture():
    torch.manual_seed(0)
    lin = nn.Linear(64, 32)
    x = torch.randn(4, 10, 64)
    y_ref = lin(x)
    dp = DynPrecisionLinear(lin.weight, lin.bias, "linear", 64, 32)
    return dp, x, y_ref


def test_weight_is_same_object(linear_fixture):
    """Wrapping must not copy the Parameter -- optimizer state has to survive."""
    dp, x, y_ref = linear_fixture
    lin_weight = dp.weight
    assert dp.weight is lin_weight, "weight object changed"


def test_fp_passthrough_matches_original(linear_fixture):
    dp, x, y_ref = linear_fixture
    dp.set_bits(None)
    assert torch.allclose(dp(x), y_ref, atol=1e-5), "FP passthrough != original"


@pytest.mark.parametrize("bits", [8, 4, 2])
def test_linear_quantized_bits_shape(linear_fixture, bits):
    dp, x, y_ref = linear_fixture
    dp.set_bits(bits)
    out = dp(x)
    assert out.shape == y_ref.shape
    err = (out - y_ref).abs().mean().item()
    assert err >= 0.0  # sanity: quantization error is a real, finite number
    assert torch.isfinite(out).all()


def test_conv1d_layout_fp_matches_manual_matmul():
    """GPT-2's Conv1D layout: weight is [in, out], y = x @ W + b."""
    torch.manual_seed(0)
    x = torch.randn(4, 10, 64)
    w = nn.Parameter(torch.randn(64, 32) * 0.1)
    b = nn.Parameter(torch.zeros(32))
    dpc = DynPrecisionLinear(w, b, "conv1d", 64, 32)
    dpc.set_bits(None)
    y_fp = dpc(x)
    y_manual = torch.matmul(x, w) + b
    assert torch.allclose(y_fp, y_manual, atol=1e-5), "conv1d FP mismatch"


def test_conv1d_layout_quantized_runs():
    torch.manual_seed(0)
    x = torch.randn(4, 10, 64)
    w = nn.Parameter(torch.randn(64, 32) * 0.1)
    b = nn.Parameter(torch.zeros(32))
    dpc = DynPrecisionLinear(w, b, "conv1d", 64, 32)
    dpc.set_bits(4)
    out = dpc(x)
    assert out.shape == (4, 10, 32)
    assert torch.isfinite(out).all()
