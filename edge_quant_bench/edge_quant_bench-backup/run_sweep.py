"""
Main sweep driver. Iterates config matrix -> results/raw/<id>.json.
Crash-safe: skips configs whose raw file already exists.
"""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))

import json, time, gc, logging, hashlib, subprocess
from pathlib import Path
import torch
import torch._dynamo
torch._dynamo.config.suppress_errors = True

# transformers 5.x output_capturing.py only imports torch under TYPE_CHECKING.
# When torch.compile inlines the BERT graph into the wrapper closure, the compiled
# code references `torch` which is not in output_capturing.py's module globals.
# Fix: inject torch into that module's __dict__ so the compiled frame can see it.
try:
    import transformers.utils.output_capturing as _oc
    import sys as _sys
    _oc.__dict__["torch"] = torch
    _oc.is_torchdynamo_compiling = lambda: torch.compiler.is_compiling()
except Exception:
    pass

import config as C
from models import get_model, get_sample_input, get_kind
from quantize import apply_scheme
from backends import make_eager, make_compiled
from measure import time_model
from deterministic import estimate_flops, weight_bytes, analytical_mem_bytes, profile_kernels
from thermal import snapshot, cooldown
from accuracy import bert_fidelity, gpt2_perplexity, vit_fidelity

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(f"{C.LOGS_DIR}/sweep.log"),
    ],
)
log = logging.getLogger("sweep")

Path(C.RAW_DIR).mkdir(parents=True, exist_ok=True)
Path(C.LOGS_DIR).mkdir(parents=True, exist_ok=True)


def get_git_commit():
    try:
        r = subprocess.run("git rev-parse --short HEAD", shell=True,
                           capture_output=True, text=True)
        return r.stdout.strip()
    except Exception:
        return "unknown"


def config_id(model, scheme, backend, batch, seq):
    s = f"{model}__{scheme}__{backend}__b{batch}__s{seq}"
    return s.replace("-", "_")


def save_result(row: dict):
    cid = config_id(row["model"], row["scheme"], row["backend"],
                    row["batch"], row["seq"])
    p = Path(C.RAW_DIR) / f"{cid}.json"
    p.write_text(json.dumps(row, indent=2, default=str))
    log.info(f"Saved {p.name}")


def result_exists(model, scheme, backend, batch, seq):
    cid = config_id(model, scheme, backend, batch, seq)
    return (Path(C.RAW_DIR) / f"{cid}.json").exists()


def get_fp16_model_ref(model_name, seq):
    """Load a fresh FP16 model for accuracy comparison."""
    m = get_model(model_name)
    return m.half()


def run_one(model_name, scheme, backend, batch, seq, fp16_ref=None, git_commit=""):
    if result_exists(model_name, scheme, backend, batch, seq):
        log.info(f"SKIP (exists): {config_id(model_name, scheme, backend, batch, seq)}")
        return

    log.info(f"START: {model_name} | {scheme} | {backend} | batch={batch} seq={seq}")
    row = dict(
        model=model_name, scheme=scheme, backend=backend,
        batch=batch, seq=seq, git_commit=git_commit,
        timestamp=time.strftime("%Y-%m-%dT%H:%M:%S"),
        oom=False, error=None,
    )

    # ── Thermal snapshot before config ──
    therm_before = snapshot()
    row["temp_start"] = therm_before["temp_c"]
    row["sm_clock_mhz"] = therm_before["sm_clock_mhz"]
    row["power_w"] = therm_before["power_w"]

    try:
        # Load model fresh
        model = get_model(model_name)
        inputs = get_sample_input(model_name, batch, seq)

        # Apply quant scheme
        model, quant_err = apply_scheme(model, scheme)
        if quant_err:
            row["error"] = quant_err
            log.warning(f"Quantization error: {quant_err}")
            # Still record deterministic metrics if model is usable
            row.update({"latency_mean_ms": -1, "latency_std_ms": -1,
                        "latency_p50_ms": -1, "latency_p90_ms": -1, "latency_p99_ms": -1})

        model.eval()

        # Cast inputs to match model dtype for non-NLP models
        if model_name in ("vit-s",):
            inp_dtype = next(model.parameters()).dtype
            if isinstance(inputs, dict):
                inputs = {k: v.to(inp_dtype) if v.is_floating_point() else v
                          for k, v in inputs.items()}
            else:
                inputs = inputs.to(inp_dtype)

        # ── Deterministic metrics (always-on) ──
        row["weight_bytes"] = weight_bytes(model)
        row["analytical_mem_bytes"] = analytical_mem_bytes(model, inputs, scheme)

        flops = estimate_flops(model, inputs)
        row["flops"] = flops

        # ── Backend wrapping ──
        compile_time = 0.0
        if backend == "eager":
            if isinstance(inputs, dict):
                def run_fn(): return model(**inputs)
            else:
                def run_fn(): return model(inputs)
        elif backend == "compile":
            torch.set_float32_matmul_precision("high")
            compiled = torch.compile(model, mode="max-autotune")
            if isinstance(inputs, dict):
                def run_fn(): return compiled(**inputs)
            else:
                def run_fn(): return compiled(inputs)
            # Warmup compile
            t0 = time.time()
            try:
                with torch.inference_mode():
                    run_fn()
                    torch.cuda.synchronize()
                compile_time = time.time() - t0
            except Exception as e:
                log.warning(f"Compile warmup error: {e}")
                compile_time = time.time() - t0
        else:
            raise ValueError(f"Unknown backend: {backend}")

        row["compile_time_s"] = compile_time

        # ── Kernel profiling ──
        try:
            kern = profile_kernels(run_fn, n_iters=3)
            row.update(kern)
        except Exception as e:
            log.warning(f"Kernel profiling failed: {e}")
            row["kernel_count"] = -1
            row["kernel_mix"] = []
            row["int8_gemm_dispatched"] = False

        # ── Latency measurement ──
        def cooldown_fn():
            t_s, t_e = cooldown(label=f"{model_name}/{scheme}/{backend}")
            row["temp_end"] = t_e

        try:
            torch.cuda.reset_peak_memory_stats()
            stats = time_model(run_fn, cooldown_fn=cooldown_fn)
            row["latency_mean_ms"] = round(stats["mean"], 4)
            row["latency_std_ms"] = round(stats["std"], 4)
            row["latency_p50_ms"] = round(stats["p50"], 4)
            row["latency_p90_ms"] = round(stats["p90"], 4)
            row["latency_p99_ms"] = round(stats["p99"], 4)
            row["latency_n"] = stats["n"]
            row["peak_mem_mb"] = round(stats["peak_mem_mb"], 2)
            row["throughput_samples_per_s"] = round(
                batch / (stats["mean"] / 1000.0), 2) if stats["mean"] > 0 else 0
            row["latency_cv"] = round(stats["std"] / stats["mean"], 4) if stats["mean"] > 0 else -1
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                log.warning(f"OOM: {e}")
                row["oom"] = True
                torch.cuda.empty_cache()
                row.update({"latency_mean_ms": -1, "latency_std_ms": -1,
                            "latency_p50_ms": -1, "latency_p90_ms": -1,
                            "latency_p99_ms": -1, "peak_mem_mb": -1,
                            "throughput_samples_per_s": 0})
            else:
                raise

        # ── Accuracy / fidelity ──
        kind = get_kind(model_name)
        acc = {"metric": "none", "flag": False}
        try:
            if kind in ("encoder",) and fp16_ref is not None and not row.get("oom"):
                acc = bert_fidelity(fp16_ref, model, inputs)
            elif kind == "decoder" and not row.get("oom"):
                acc = gpt2_perplexity(fp16_ref, model)
            elif kind == "vit" and fp16_ref is not None and not row.get("oom"):
                acc = vit_fidelity(fp16_ref, model, inputs)
        except Exception as e:
            log.warning(f"Accuracy check failed: {e}")
            acc = {"metric": "error", "flag": True, "error": str(e)}
        row["accuracy"] = acc

    except RuntimeError as e:
        if "out of memory" in str(e).lower():
            log.warning(f"OOM during setup: {e}")
            row["oom"] = True
            torch.cuda.empty_cache()
            row.setdefault("latency_mean_ms", -1)
        else:
            log.error(f"Error in run_one: {e}", exc_info=True)
            row["error"] = str(e)
    except Exception as e:
        log.error(f"Unexpected error: {e}", exc_info=True)
        row["error"] = str(e)
    finally:
        therm_after = snapshot()
        row.setdefault("temp_end", therm_after["temp_c"])
        # Cleanup
        try:
            del model
        except Exception:
            pass
        gc.collect()
        torch.cuda.empty_cache()

    save_result(row)


def run_phase1(models=None, schemes=None, backends=None, batches=None, seq=C.SEQ_LEN_DEFAULT):
    git_commit = get_git_commit()
    models = models or C.MODELS_PHASE1
    schemes = schemes or C.SCHEMES_PHASE1
    backends = backends or C.BACKENDS_PHASE1
    batches = batches or C.BATCHES

    total = len(models) * len(schemes) * len(backends) * len(batches)
    done = 0

    for model_name in models:
        # Load FP16 ref once per model (for accuracy comparison)
        log.info(f"Loading FP16 reference for {model_name}")
        try:
            fp16_ref = get_fp16_model_ref(model_name, seq)
            fp16_ref.eval()
        except Exception as e:
            log.warning(f"Could not load FP16 ref for {model_name}: {e}")
            fp16_ref = None

        for scheme in schemes:
            for backend in backends:
                for batch in batches:
                    done += 1
                    log.info(f"Progress: {done}/{total}")
                    run_one(model_name, scheme, backend, batch, seq,
                            fp16_ref=fp16_ref, git_commit=git_commit)

        del fp16_ref
        gc.collect()
        torch.cuda.empty_cache()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", nargs="+", default=None)
    parser.add_argument("--schemes", nargs="+", default=None)
    parser.add_argument("--backends", nargs="+", default=None)
    parser.add_argument("--batches", nargs="+", type=int, default=None)
    parser.add_argument("--seq", type=int, default=C.SEQ_LEN_DEFAULT)
    parser.add_argument("--phase0", action="store_true",
                        help="Phase 0 sanity run: bert-base, {fp32,fp16}, {eager,compile}, batch{1,16}")
    args = parser.parse_args()

    if args.phase0:
        log.info("=== Phase 0 sanity run ===")
        run_phase1(
            models=["bert-base"],
            schemes=["fp32", "fp16"],
            backends=["eager", "compile"],
            batches=[1, 16],
            seq=128,
        )
    else:
        run_phase1(
            models=args.models,
            schemes=args.schemes,
            backends=args.backends,
            batches=args.batches,
            seq=args.seq,
        )
