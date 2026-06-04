"""
Cheap always-on metrics: kernel count/mix, FLOPs, weight bytes, analytical mem.
NOTE: peak_mem_mb (from torch.cuda.max_memory_allocated) is RESIDENT FOOTPRINT,
not DRAM traffic. True memory traffic requires Nsight Compute (profile_ncu.py).
This distinction is explicitly documented here and in README.
"""
import torch
from torch.profiler import profile, ProfilerActivity
import logging

log = logging.getLogger(__name__)

# ── FLOPs ──────────────────────────────────────────────────────────────────────

def estimate_flops(model, inputs):
    """Analytical FLOPs via torch.profiler flop counter. Returns total FLOPs (int)."""
    try:
        with torch.inference_mode():
            with profile(
                activities=[ProfilerActivity.CPU],
                with_flops=True,
                record_shapes=True,
            ) as prof:
                if isinstance(inputs, dict):
                    model(**inputs)
                else:
                    model(inputs)
        total = sum(e.flops for e in prof.key_averages() if e.flops > 0)
        return int(total)
    except Exception as e:
        log.warning(f"FLOPs estimation failed: {e}")
        return -1


# ── Weight bytes ───────────────────────────────────────────────────────────────

def weight_bytes(model):
    """
    Post-quantization effective weight size in bytes.
    torchao stores quantized weights as packed int tensors; we read .nbytes directly.
    """
    total = 0
    for p in model.parameters():
        total += p.nbytes
    # Also count buffers (torchao may store scales/zeros as buffers)
    for b in model.buffers():
        total += b.nbytes
    return total


# ── Analytical memory traffic model ───────────────────────────────────────────

def analytical_mem_bytes(model, inputs, scheme):
    """
    Approximate DRAM traffic = weight bytes + activation bytes per forward pass.
    This is an ANALYTICAL MODEL, not measured. Label as such in figures.
    Activation bytes estimated as sum of parameter shapes' output activations.
    """
    w_bytes = weight_bytes(model)
    # Rough activation estimate: same order as weight bytes for typical transformers
    # (each layer reads weights once + reads/writes activations once)
    # We use 2× weight bytes as a first-order estimate for total traffic.
    # For CNNs the ratio differs but this gives a consistent relative comparison.
    return w_bytes * 2  # approximate; see README


# ── Kernel profiling ───────────────────────────────────────────────────────────

def profile_kernels(run_fn, n_iters=3):
    """
    Run run_fn n_iters times under CUDA profiler. Return kernel stats dict.
    Returns: {kernel_count, kernel_mix: [{name, count, cuda_time_ms}], int8_gemm_dispatched}
    """
    try:
        with torch.inference_mode():
            with profile(
                activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
                record_shapes=False,
            ) as prof:
                for _ in range(n_iters):
                    run_fn()
                    torch.cuda.synchronize()

        avgs = prof.key_averages()
        kernel_mix = []
        for e in avgs:
            # PyTorch 2.4+ renamed self_cuda_time_total -> self_device_time_total
            t = getattr(e, "self_device_time_total",
                        getattr(e, "self_cuda_time_total", 0))
            if t > 0:
                kernel_mix.append({
                    "name": e.key,
                    "count": e.count,
                    "cuda_time_ms": round(t / 1000.0, 4),
                })

        kernel_mix.sort(key=lambda x: x["cuda_time_ms"], reverse=True)
        kernel_count = sum(k["count"] for k in kernel_mix)

        # H2 test: did a real INT8/cutlass GEMM dispatch?
        # Match INT8-specific GEMM patterns. "sm89" alone is just the arch tag and
        # appears in FP16 kernels too, so require it paired with "i8"/"int8" patterns.
        int8_patterns = [
            "gemm_i8", "int8_gemm", "i8i8", "int8_weight", "i8_i8",
            "tensorop_i8816", "volta_i8", "turing_i8", "ampere_i8",
            "cutlass_i8", "xmma_gemm_i8",
        ]
        int8_gemm_dispatched = any(
            any(pat in k["name"].lower() for pat in int8_patterns)
            for k in kernel_mix
        )

        return {
            "kernel_count": kernel_count,
            "kernel_mix": kernel_mix[:20],  # top-20 by cuda time
            "int8_gemm_dispatched": int8_gemm_dispatched,
        }
    except Exception as e:
        log.warning(f"Kernel profiling failed: {e}")
        return {
            "kernel_count": -1,
            "kernel_mix": [],
            "int8_gemm_dispatched": False,
        }
