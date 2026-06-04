# src/evaluation/profiler.py

import os
import json
import torch
import torch.nn as nn
from torch.profiler import profile, ProfilerActivity, record_function
from typing import Dict


def profile_model(
    model: nn.Module,
    inputs: dict,
    warmup_runs: int = 5,
    profile_runs: int = 10,
    label: str = "model",
) -> Dict:
    model.eval()
    device = next(model.parameters()).device
    inputs = {k: v.to(device) for k, v in inputs.items()}

    print(f"    [{label}] warming up ({warmup_runs} runs)...")
    with torch.no_grad():
        for _ in range(warmup_runs):
            model(**inputs)
    torch.cuda.synchronize()

    print(f"    [{label}] profiling ({profile_runs} runs)...")
    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        record_shapes=False,
        profile_memory=False,
    ) as prof:
        with torch.no_grad():
            for _ in range(profile_runs):
                with record_function(label):
                    model(**inputs)
    torch.cuda.synchronize()

    key_averages = prof.key_averages()

    # use device_time_total — works across PyTorch versions
    cuda_events  = [e for e in key_averages if e.device_time_total > 0]
    kernel_count = len(cuda_events)

    total_cuda_us = sum(e.device_time_total for e in key_averages)
    total_cuda_ms = round(total_cuda_us / 1000 / profile_runs, 4)

    top_ops = []
    for e in sorted(cuda_events, key=lambda e: e.device_time_total, reverse=True)[:20]:
        top_ops.append({
            "name":         e.key,
            "cuda_time_ms": round(e.device_time_total / 1000 / profile_runs, 5),
            "count":        e.count,
        })

    return {
        "label":              label,
        "kernel_count":       kernel_count,
        "total_cuda_time_ms": total_cuda_ms,
        "top_ops":            top_ops,
    }


def measure_latency(
    model: nn.Module,
    inputs: dict,
    warmup_runs: int = 20,
    timed_runs:  int = 100,
    label: str = "model",
) -> Dict:
    model.eval()
    device = next(model.parameters()).device
    inputs = {k: v.to(device) for k, v in inputs.items()}

    print(f"    [{label}] warming up ({warmup_runs} runs)...")
    with torch.no_grad():
        for _ in range(warmup_runs):
            model(**inputs)
    torch.cuda.synchronize()

    print(f"    [{label}] timing ({timed_runs} runs)...")
    latencies = []
    with torch.no_grad():
        for _ in range(timed_runs):
            t0 = torch.cuda.Event(enable_timing=True)
            t1 = torch.cuda.Event(enable_timing=True)
            t0.record()
            model(**inputs)
            t1.record()
            torch.cuda.synchronize()
            latencies.append(t0.elapsed_time(t1))

    t = torch.tensor(latencies)
    return {
        "label":   label,
        "mean_ms": round(t.mean().item(), 4),
        "std_ms":  round(t.std().item(),  4),
        "min_ms":  round(t.min().item(),  4),
        "max_ms":  round(t.max().item(),  4),
    }


def save_results(data: Dict, folder: str, filename: str):
    os.makedirs(folder, exist_ok=True)
    path = os.path.join(folder, filename)
    with open(path, "w") as f:
        json.dump(data, f, indent=2)
    print(f"    Saved → {path}")


def print_comparison(fp32_prof, qat_prof, fp32_lat, qat_lat):
    print("\n" + "="*62)
    print("  RESULTS")
    print("="*62)
    print(f"  {'Metric':<32} {'FP32':>10} {'Std QAT':>12}")
    print("  " + "-"*58)
    print(f"  {'CUDA kernel count':<32} "
          f"{fp32_prof['kernel_count']:>10} "
          f"{qat_prof['kernel_count']:>12}")
    print(f"  {'Total CUDA time (ms)':<32} "
          f"{fp32_prof['total_cuda_time_ms']:>10} "
          f"{qat_prof['total_cuda_time_ms']:>12}")
    print(f"  {'Latency mean (ms)':<32} "
          f"{fp32_lat['mean_ms']:>10} "
          f"{qat_lat['mean_ms']:>12}")
    print(f"  {'Latency std (ms)':<32} "
          f"{fp32_lat['std_ms']:>10} "
          f"{qat_lat['std_ms']:>12}")
    print("  " + "-"*58)

    kernel_ratio = qat_prof['kernel_count'] / max(fp32_prof['kernel_count'], 1)
    speedup      = fp32_lat['mean_ms'] / max(qat_lat['mean_ms'], 1e-9)
    print(f"  {'QAT kernel increase':<32} {kernel_ratio:>10.2f}x")
    print(f"  {'QAT speedup over FP32':<32} {speedup:>12.2f}x")
    print("="*62)

    print("\n  QAT top ops (QDQ ops marked):")
    print(f"  {'Op':<52} {'ms':>8}")
    print("  " + "-"*62)
    for op in qat_prof["top_ops"][:12]:
        tag = " ◄ QDQ" if any(
            k in op["name"].lower()
            for k in ["quant", "dequant", "fake", "activation_post_process"]
        ) else ""
        print(f"  {op['name'][:52]:<52} {op['cuda_time_ms']:>8.5f}{tag}")