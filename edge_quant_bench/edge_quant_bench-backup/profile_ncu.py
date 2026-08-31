"""
Nsight Compute subset: DRAM traffic, occupancy, SM throughput.
Run on representative ~6-10 cells only (slow; may need elevated permissions).
Falls back to analytical model if ncu unavailable.
"""
import subprocess, json, os, tempfile, logging
from pathlib import Path

log = logging.getLogger(__name__)


def ncu_available():
    r = subprocess.run("which ncu && ncu --version", shell=True, capture_output=True, text=True)
    return r.returncode == 0 and "version" in r.stdout.lower()


def run_ncu_on_script(script_path: str, out_json: str, extra_env: dict = None):
    """
    Run ncu on a self-contained Python script, parse output to JSON.
    Captures: dram__bytes_read, dram__bytes_write, achieved_occupancy, sm__throughput.
    """
    if not ncu_available():
        log.warning("ncu not available; skipping Nsight Compute profile")
        return None

    ncu_out = out_json.replace(".json", ".ncu-rep")

    metrics = (
        "dram__bytes_read.sum,"
        "dram__bytes_write.sum,"
        "sm__throughput.avg.pct_of_peak_sustained_elapsed,"
        "l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum,"
        "l1tex__t_sectors_pipe_lsu_mem_global_op_st.sum,"
        "sm__sass_thread_inst_executed_op_fadd_pred_on.sum,"
        "sm__sass_thread_inst_executed_op_fmul_pred_on.sum,"
        "sm__sass_thread_inst_executed_op_ffma_pred_on.sum"
    )

    python_exe = "/home/shreya/venvs/fusion_qat/bin/python"
    cmd = (
        f"ncu --metrics {metrics} --csv "
        f"--target-processes all "
        f"{python_exe} {script_path}"
    )

    env = os.environ.copy()
    if extra_env:
        env.update(extra_env)

    log.info(f"Running ncu: {cmd}")
    r = subprocess.run(cmd, shell=True, capture_output=True, text=True,
                       timeout=300, env=env)

    if r.returncode != 0:
        log.warning(f"ncu failed (code {r.returncode}): {r.stderr[:500]}")
        return {"error": r.stderr[:500], "stdout": r.stdout[:500]}

    # Parse CSV output
    lines = [l for l in r.stdout.split("\n") if l.strip() and not l.startswith("==")]
    results = {"raw_stdout": r.stdout[:2000], "metrics": {}}

    for line in lines:
        parts = line.split(",")
        if len(parts) >= 5:
            metric_name = parts[2].strip().strip('"')
            value_str = parts[4].strip().strip('"')
            try:
                results["metrics"][metric_name] = float(value_str.replace(",", ""))
            except ValueError:
                results["metrics"][metric_name] = value_str

    Path(out_json).write_text(json.dumps(results, indent=2))
    return results


def analytical_roofline(weight_bytes: int, flops: int, dtype_bytes: int = 2):
    """
    Compute arithmetic intensity (FLOPs/byte) from analytical traffic model.
    Used as fallback when ncu is unavailable.
    """
    # Analytical traffic: weight bytes + rough activation estimate
    traffic_bytes = weight_bytes * 2
    if traffic_bytes == 0:
        return 0.0
    ai = flops / traffic_bytes
    return round(ai, 4)
