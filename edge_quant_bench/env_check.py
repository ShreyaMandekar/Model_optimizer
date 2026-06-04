"""Environment check — run before anything else."""
import json, subprocess, sys, os
from pathlib import Path

def _ver(pkg):
    try:
        import importlib.metadata as im
        return im.version(pkg)
    except Exception:
        return "not installed"

def _run(cmd):
    try:
        r = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=10)
        return r.stdout.strip()
    except Exception as e:
        return f"error: {e}"

def main():
    import torch

    if not torch.cuda.is_available():
        print("FATAL: torch.cuda.is_available() is False. Cannot proceed.")
        sys.exit(1)

    gpu = torch.cuda.get_device_properties(0)
    free, total = torch.cuda.mem_get_info(0)

    sudo_ok = _run("sudo -n nvidia-smi -L 2>&1").startswith("GPU")

    ncu_ver = _run("ncu --version 2>&1")
    ncu_available = "version" in ncu_ver.lower() or "CUDA" in ncu_ver

    info = {
        "python_exe": sys.executable,
        "python_version": sys.version,
        "torch": torch.__version__,
        "cuda_version": torch.version.cuda,
        "torchao": _ver("torchao"),
        "transformers": _ver("transformers"),
        "datasets": _ver("datasets"),
        "onnxruntime_gpu": _ver("onnxruntime-gpu"),
        "timm": _ver("timm"),
        "gpu_name": gpu.name,
        "gpu_total_vram_gb": round(gpu.total_memory / 1e9, 2),
        "gpu_free_vram_gb": round(free / 1e9, 2),
        "gpu_sm_count": gpu.multi_processor_count,
        "gpu_compute_capability": f"{gpu.major}.{gpu.minor}",
        "sm_clock_mhz": _run("nvidia-smi --query-gpu=clocks.sm --format=csv,noheader,nounits").split("\n")[0].strip(),
        "gpu_temp_c": _run("nvidia-smi --query-gpu=temperature.gpu --format=csv,noheader,nounits").split("\n")[0].strip(),
        "sudo_nvidia_smi": sudo_ok,
        "ncu_available": ncu_available,
        "ncu_version": ncu_ver[:200],
    }

    for k, v in info.items():
        print(f"  {k}: {v}")

    out = Path(__file__).parent / "results" / "env.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(info, indent=2))
    print(f"\nSaved to {out}")
    return info

if __name__ == "__main__":
    main()
