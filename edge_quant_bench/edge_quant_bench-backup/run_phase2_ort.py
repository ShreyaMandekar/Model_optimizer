"""
Phase 2: ONNX Runtime (CUDA EP) sweep for all models including ResNet-50.
Exports models to ONNX, runs with ORT FP32/FP16/INT8.
"""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))

import json, time, gc, logging, subprocess
from pathlib import Path
import torch
import numpy as np
import torch._dynamo
torch._dynamo.config.suppress_errors = True

try:
    import transformers.utils.output_capturing as _oc
    _oc.__dict__["torch"] = torch
    _oc.is_torchdynamo_compiling = lambda: torch.compiler.is_compiling()
except Exception:
    pass

import config as C
from models import get_model, get_sample_input, get_kind
from thermal import snapshot, cooldown
from deterministic import weight_bytes
import statistics as st

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
                    handlers=[logging.StreamHandler(),
                               logging.FileHandler(f"{C.LOGS_DIR}/phase2.log")])
log = logging.getLogger("phase2")

Path(C.RAW_DIR).mkdir(parents=True, exist_ok=True)
ONNX_DIR = Path("onnx_models")
ONNX_DIR.mkdir(exist_ok=True)


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


def result_exists(model, scheme, backend, batch, seq):
    cid = config_id(model, scheme, backend, batch, seq)
    return (Path(C.RAW_DIR) / f"{cid}.json").exists()


def save_result(row):
    cid = config_id(row["model"], row["scheme"], row["backend"],
                    row["batch"], row["seq"])
    p = Path(C.RAW_DIR) / f"{cid}.json"
    p.write_text(json.dumps(row, indent=2, default=str))
    log.info(f"Saved {p.name}")


def export_onnx(model_name, model, inputs_dict, seq, batch):
    onnx_path = ONNX_DIR / f"{model_name.replace('-','_')}_b{batch}_s{seq}.onnx"
    if onnx_path.exists():
        return str(onnx_path), None
    try:
        import warnings
        with torch.inference_mode(), warnings.catch_warnings():
            warnings.simplefilter("ignore")
            if isinstance(inputs_dict, dict):
                input_names = list(inputs_dict.keys())
                # Wrap to avoid transformers 5.x use_cache / DynamicCache conflicts
                class _Wrapper(torch.nn.Module):
                    def __init__(self, m): super().__init__(); self.m = m
                    def forward(self, *args):
                        kw = dict(zip(input_names, args))
                        kw["use_cache"] = False  # disable KV cache so no DynamicCache output
                        return self.m(**kw)
                wrapper = _Wrapper(model)
                args = tuple(inputs_dict.values())
                torch.onnx.export(
                    wrapper, args,
                    str(onnx_path),
                    input_names=input_names,
                    opset_version=17,
                    do_constant_folding=True,
                )
            else:
                torch.onnx.export(
                    model, (inputs_dict,),
                    str(onnx_path),
                    opset_version=17,
                    do_constant_folding=True,
                )
        log.info(f"Exported ONNX: {onnx_path}")
        return str(onnx_path), None
    except Exception as e:
        return None, str(e)


def run_ort(onnx_path, inputs_np, batch, warmup=10, iters=50):
    """Run ORT inference and return latency stats."""
    import onnxruntime as ort
    try:
        sess_opts = ort.SessionOptions()
        sess_opts.log_severity_level = 3
        sess = ort.InferenceSession(
            onnx_path,
            sess_options=sess_opts,
            providers=["CUDAExecutionProvider", "CPUExecutionProvider"]
        )
        # Build feed dict: dict inputs pass as-is; plain arrays use model's input name
        if isinstance(inputs_np, dict):
            feed = inputs_np
        else:
            feed = {sess.get_inputs()[0].name: inputs_np}

        def run():
            return sess.run(None, feed)

        # Warmup
        for _ in range(warmup):
            run()

        # Measure
        all_t = []
        for _ in range(iters):
            t0 = time.perf_counter()
            run()
            all_t.append((time.perf_counter() - t0) * 1000)  # ms

        all_t.sort()
        n = len(all_t)
        pct = lambda p: all_t[min(n-1, int(n*p/100))]
        return {
            "mean": st.mean(all_t), "std": st.pstdev(all_t),
            "p50": pct(50), "p90": pct(90), "p99": pct(99),
            "n": n,
        }, None
    except Exception as e:
        return None, str(e)


def to_numpy(inputs, dtype_fp16=False):
    """Convert torch tensors to numpy for ORT."""
    if isinstance(inputs, dict):
        result = {}
        for k, v in inputs.items():
            if v.dtype == torch.long or v.dtype == torch.int:
                result[k] = v.cpu().numpy().astype(np.int64)
            elif dtype_fp16:
                result[k] = v.cpu().float().numpy().astype(np.float16)
            else:
                result[k] = v.cpu().float().numpy()
        return result
    else:
        if dtype_fp16:
            return inputs.cpu().float().numpy().astype(np.float16)
        return inputs.cpu().float().numpy()


def run_phase2(models=None, batches=None, seq=C.SEQ_LEN_DEFAULT):
    git_commit = get_git_commit()
    all_models = (models or C.MODELS_PHASE1) + C.MODELS_PHASE2
    batches = batches or C.BATCHES

    for model_name in all_models:
        log.info(f"\n=== Phase 2 ORT: {model_name} ===")
        for batch in batches:
            if result_exists(model_name, "fp32", "ort", batch, seq):
                log.info(f"SKIP: {model_name} ort batch={batch}")
                continue

            row_base = dict(
                model=model_name, backend="ort",
                batch=batch, seq=seq, git_commit=git_commit,
                timestamp=time.strftime("%Y-%m-%dT%H:%M:%S"),
            )

            # Load FP32 model for export
            try:
                model = get_model(model_name).float()
                inputs = get_sample_input(model_name, batch, seq)
            except Exception as e:
                log.error(f"Failed to load {model_name}: {e}")
                continue

            # Export ONNX (FP32 base)
            onnx_path, err = export_onnx(model_name, model, inputs, seq, batch)
            if err:
                log.warning(f"ONNX export failed: {err}")
                row = {**row_base, "scheme": "fp32", "error": err}
                save_result(row)
                del model; gc.collect(); torch.cuda.empty_cache()
                continue

            inputs_np = to_numpy(inputs)

            # Run ORT FP32
            therm = snapshot()
            stats, err = run_ort(onnx_path, inputs_np, batch)
            if err:
                log.warning(f"ORT FP32 failed: {err}")
            row = {**row_base, "scheme": "fp32",
                   "temp_start": therm["temp_c"], "sm_clock_mhz": therm["sm_clock_mhz"],
                   "weight_bytes": weight_bytes(model), "error": err}
            if stats:
                row.update({
                    "latency_mean_ms": round(stats["mean"], 4),
                    "latency_std_ms": round(stats["std"], 4),
                    "latency_p50_ms": round(stats["p50"], 4),
                    "latency_p90_ms": round(stats["p90"], 4),
                    "latency_p99_ms": round(stats["p99"], 4),
                    "throughput_samples_per_s": round(batch / (stats["mean"] / 1000), 2) if stats["mean"] > 0 else 0,
                })
            save_result(row)

            # Run ORT FP16 (cast input to fp16 if float)
            inputs_np_fp16 = to_numpy(inputs, dtype_fp16=True)
            therm = snapshot()
            stats16, err16 = run_ort(onnx_path, inputs_np_fp16, batch)
            row16 = {**row_base, "scheme": "fp16",
                     "temp_start": therm["temp_c"], "sm_clock_mhz": therm["sm_clock_mhz"],
                     "weight_bytes": weight_bytes(model), "error": err16}
            if stats16:
                row16.update({
                    "latency_mean_ms": round(stats16["mean"], 4),
                    "latency_std_ms": round(stats16["std"], 4),
                    "latency_p50_ms": round(stats16["p50"], 4),
                    "latency_p90_ms": round(stats16["p90"], 4),
                    "latency_p99_ms": round(stats16["p99"], 4),
                    "throughput_samples_per_s": round(batch / (stats16["mean"] / 1000), 2) if stats16["mean"] > 0 else 0,
                })
            save_result(row16)

            del model; gc.collect(); torch.cuda.empty_cache()
            cooldown(label=f"{model_name}/ort/batch={batch}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", nargs="+", default=None)
    parser.add_argument("--batches", nargs="+", type=int, default=None)
    parser.add_argument("--seq", type=int, default=C.SEQ_LEN_DEFAULT)
    args = parser.parse_args()
    run_phase2(models=args.models, batches=args.batches, seq=args.seq)
