"""Wrap quantized model as a callable: eager / compile / onnxruntime."""
import torch
import time
import logging

log = logging.getLogger(__name__)


def make_eager(model, inputs):
    """Return (run_fn, compile_time_s). Eager = no torch.compile."""
    def run():
        return model(**inputs) if isinstance(inputs, dict) else model(inputs)
    return run, 0.0


def make_compiled(model, inputs):
    """Compile the model with torch.compile, return (run_fn, compile_time_s)."""
    torch.set_float32_matmul_precision("high")

    if isinstance(inputs, dict):
        def run():
            return model(**inputs)
    else:
        def run():
            return model(inputs)

    t0 = time.time()
    try:
        compiled = torch.compile(model, mode="max-autotune")
        # Replace model reference
        if isinstance(inputs, dict):
            def run():
                return compiled(**inputs)
        else:
            def run():
                return compiled(inputs)

        # Trigger compile by running once
        with torch.inference_mode():
            run()
            torch.cuda.synchronize()
        compile_time = time.time() - t0
    except Exception as e:
        log.warning(f"torch.compile failed ({e}), falling back to eager")
        compile_time = time.time() - t0
        # run already defined as eager above

    return run, compile_time


def make_ort(model_path: str, inputs_np: dict, providers=None):
    """Wrap an ONNX model with ORT CUDA EP."""
    import onnxruntime as ort
    if providers is None:
        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
    sess = ort.InferenceSession(model_path, providers=providers)

    def run():
        return sess.run(None, inputs_np)

    return run, 0.0
