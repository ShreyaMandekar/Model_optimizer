"""Timing harness: latency stats, throughput, peak memory."""
import torch
import statistics as st
from config import WARMUP_ITERS, MEASURE_BLOCKS, MEASURE_ITERS_PER_BLOCK


def time_model(run_fn, warmup=WARMUP_ITERS, blocks=MEASURE_BLOCKS,
               iters=MEASURE_ITERS_PER_BLOCK, cooldown_fn=None):
    """
    Returns dict with mean, std, p50, p90, p99 (ms), n, peak_mem_mb.
    run_fn must be a zero-arg callable that calls the model synchronously.
    """
    torch.cuda.reset_peak_memory_stats()

    with torch.inference_mode():
        # Warmup (triggers compile + autotune)
        for _ in range(warmup):
            run_fn()
        torch.cuda.synchronize()

        all_t = []
        for _ in range(blocks):
            for _ in range(iters):
                s = torch.cuda.Event(enable_timing=True)
                e = torch.cuda.Event(enable_timing=True)
                s.record()
                run_fn()
                e.record()
                torch.cuda.synchronize()
                all_t.append(s.elapsed_time(e))  # ms
            if cooldown_fn is not None:
                cooldown_fn()

    all_t.sort()
    n = len(all_t)

    def pct(p):
        return all_t[min(n - 1, int(n * p / 100))]

    peak_mem = torch.cuda.max_memory_allocated() / 1e6  # bytes -> MB

    return dict(
        mean=st.mean(all_t),
        std=st.pstdev(all_t),
        p50=pct(50),
        p90=pct(90),
        p99=pct(99),
        n=n,
        peak_mem_mb=peak_mem,
    )
