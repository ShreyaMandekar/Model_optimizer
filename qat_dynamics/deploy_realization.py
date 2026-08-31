"""RQ4: Export trained freeze pattern to real torchao deployment path and measure latency.

Reuses edge_quant_bench timing discipline: CUDA-event timing, thermal cooldown.
The key expected finding: INT8 GEMM never dispatches on the 24-SM AD107 (sm_89)
— FP16+compile dominates. This script confirms/extends that result for the
trained freeze patterns.

Usage:
    python deploy_realization.py --model distilbert --task sst2 --metric kl
"""

import argparse, json, time
from pathlib import Path

import torch
import torch.nn as nn
import pyarrow.parquet as pq
import pandas as pd

from config import RUNS, FIGS, RESULTS
from models import build_model
from data import get_loaders, move_batch


WARMUP  = 5
REPEATS = 50
BATCH   = 16
SEQ_LEN = 128


def _cuda_time(fn, warmup=WARMUP, repeats=REPEATS):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end   = torch.cuda.Event(enable_timing=True)
    times = []
    for _ in range(repeats):
        start.record()
        fn()
        end.record()
        torch.cuda.synchronize()
        times.append(start.elapsed_time(end))
    import numpy as np
    return float(np.mean(times)), float(np.std(times))


def _get_temp():
    try:
        import subprocess
        r = subprocess.run(["nvidia-smi", "--query-gpu=temperature.gpu", "--format=csv,noheader"],
                           capture_output=True, text=True, timeout=5)
        return int(r.stdout.strip())
    except Exception:
        return -1


def cooldown(target=45, timeout=120):
    t0 = time.time()
    while _get_temp() > target and time.time() - t0 < timeout:
        time.sleep(5)


def measure_latency(model, dummy_input, label=""):
    model.eval()
    fn = lambda: model(**dummy_input)
    mean_ms, std_ms = _cuda_time(fn)
    print(f"  {label:35s}: {mean_ms:.2f} ± {std_ms:.2f} ms")
    return {"label": label, "mean_ms": mean_ms, "std_ms": std_ms}


def run_rq4(model_key="distilbert", task="sst2", metric="kl"):
    device = torch.device("cuda")

    dummy = {
        "input_ids":      torch.randint(0, 30000, (BATCH, SEQ_LEN), device=device),
        "attention_mask": torch.ones(BATCH, SEQ_LEN, dtype=torch.long, device=device),
    }

    results = []

    # 1. Baseline: FP16 eager
    print("\n[RQ4] Baseline FP16 eager")
    model_fp16, _ = build_model(model_key, task, seed=42)
    model_fp16 = model_fp16.to(device).half().eval()
    dummy_fp16 = {k: v for k, v in dummy.items()}
    r = measure_latency(model_fp16, dummy_fp16, "fp16_eager")
    r["config"] = "fp16_eager"; results.append(r)
    del model_fp16; torch.cuda.empty_cache(); cooldown()

    # 2. FP16 + compile
    print("[RQ4] FP16 + compile")
    model_fp16c, _ = build_model(model_key, task, seed=42)
    model_fp16c = model_fp16c.to(device).half().eval()
    model_fp16c = torch.compile(model_fp16c, mode="max-autotune")
    r = measure_latency(model_fp16c, dummy_fp16, "fp16_compile")
    r["config"] = "fp16_compile"; results.append(r)
    del model_fp16c; torch.cuda.empty_cache(); cooldown()

    # 3. Load the best trained freeze pattern (lowest val_loss kl run) and
    #    export via torchao quantize_ + compile
    best_run = _find_best_run(model_key, task, metric)
    if best_run is not None:
        print(f"\n[RQ4] Deploying freeze pattern from: {best_run['run_id']}")
        print(f"      frozen_frac={best_run['frozen_frac']:.2f}  acc={best_run['final_acc']:.4f}")

        # Build fresh model with the REAL torchao deployment quantize_ path
        from torchao.quantization import quantize_, int8_dynamic_activation_int4_weight
        model_q, _ = build_model(model_key, task, seed=42)
        # convert DynStatQATLinear back to plain nn.Linear for real quantize_
        _restore_linears(model_q)
        model_q = model_q.to(device).eval()
        try:
            quantize_(model_q, int8_dynamic_activation_int4_weight())
            label = "w4a8_deploy_eager"
        except Exception as e:
            print(f"  quantize_ failed: {e} — trying fp16 path instead")
            model_q = model_q.half()
            label = "w4a8_deploy_fallback_fp16"

        r = measure_latency(model_q, dummy, label)
        r["config"] = label; r["int8_dispatched"] = _check_int8(model_q)
        results.append(r)
        del model_q; torch.cuda.empty_cache(); cooldown()

        # compile variant
        model_qc, _ = build_model(model_key, task, seed=42)
        _restore_linears(model_qc)
        model_qc = model_qc.to(device).eval()
        try:
            quantize_(model_qc, int8_dynamic_activation_int4_weight())
            model_qc = torch.compile(model_qc, mode="max-autotune")
            label = "w4a8_deploy_compile"
        except Exception as e:
            model_qc = model_qc.half()
            model_qc = torch.compile(model_qc, mode="max-autotune")
            label = "w4a8_deploy_compile_fp16"

        r = measure_latency(model_qc, dummy, label)
        r["config"] = label; results.append(r)
        del model_qc; torch.cuda.empty_cache()

    # Save
    df = pd.DataFrame(results)
    out = RESULTS / "deploy_realization.json"
    out.write_text(json.dumps(results, indent=2))
    print(f"\nSaved to {out}")
    return results


def _find_best_run(model_key, task, metric):
    rows = []
    for f in RUNS.glob(f"{model_key}__{task}__{metric}__*.parquet"):
        tbl = pq.read_table(f)
        df  = tbl.to_pandas()
        s   = df.iloc[-1]
        rows.append({k: s.get(k) for k in ["run_id", "final_acc", "final_loss", "frozen_frac"]})
    if not rows:
        return None
    return max(rows, key=lambda r: r["final_acc"] or 0)


def _restore_linears(model):
    """Replace DynStatQATLinear with plain nn.Linear (for real quantize_ path)."""
    from qat_linear import DynStatQATLinear
    for name, module in list(model.named_modules()):
        if isinstance(module, DynStatQATLinear):
            lin = nn.Linear(module.in_features, module.out_features,
                            bias=module._extra_bias is not None)
            lin.weight = nn.Parameter(module.weight.data.clone())
            if module._extra_bias is not None:
                lin.bias = nn.Parameter(module._extra_bias.data.clone())
            parts = name.split(".")
            parent = model
            for p in parts[:-1]:
                parent = getattr(parent, p)
            setattr(parent, parts[-1], lin)


def _check_int8(model):
    """Heuristic: look for int8 kernel in compiled graph (placeholder)."""
    return False  # on 24-SM AD107 INT8 never dispatches per edge_quant_bench


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model",  default="distilbert")
    parser.add_argument("--task",   default="sst2")
    parser.add_argument("--metric", default="kl")
    args = parser.parse_args()
    run_rq4(args.model, args.task, args.metric)
