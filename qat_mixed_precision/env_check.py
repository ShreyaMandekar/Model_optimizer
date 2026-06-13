"""Environment verification — prints + saves results/env.json. Stops if CUDA unavailable."""
import json, os, sys
from pathlib import Path

def main():
    import torch
    if not torch.cuda.is_available():
        print("ERROR: torch CUDA unavailable. Stopping.")
        sys.exit(1)

    import torchao, transformers, datasets

    gpu = torch.cuda.get_device_properties(0)
    free, total = torch.cuda.mem_get_info(0)

    info = {
        "python": sys.version,
        "torch": torch.__version__,
        "torchao": torchao.__version__,
        "transformers": transformers.__version__,
        "datasets": datasets.__version__,
        "cuda": torch.version.cuda,
        "gpu_name": gpu.name,
        "gpu_total_mb": round(total / 1024**2, 1),
        "gpu_free_mb": round(free / 1024**2, 1),
        "sm_count": gpu.multi_processor_count,
        "sm_clock_mhz": _get_sm_clock(),
        "temperature_c": _get_temp(),
    }

    for k, v in info.items():
        print(f"  {k}: {v}")

    out = Path(__file__).parent / "results" / "env.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(info, indent=2))
    print(f"\nSaved to {out}")
    return info

def _get_sm_clock():
    try:
        import subprocess
        r = subprocess.run(
            ["nvidia-smi", "--query-gpu=clocks.sm", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=5)
        return r.stdout.strip()
    except Exception:
        return None

def _get_temp():
    try:
        import subprocess
        r = subprocess.run(
            ["nvidia-smi", "--query-gpu=temperature.gpu", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=5)
        return int(r.stdout.strip())
    except Exception:
        return None

if __name__ == "__main__":
    main()
