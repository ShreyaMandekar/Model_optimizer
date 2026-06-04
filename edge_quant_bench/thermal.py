"""Thermal management: clock lock (if sudo), temp polling, cooldown."""
import subprocess, time, logging
from config import COOLDOWN_TARGET_TEMP_C, COOLDOWN_MAX_WAIT_S

log = logging.getLogger(__name__)

def _smi(query, units=False):
    fmt = "csv,noheader" + (",nounits" if units else "")
    r = subprocess.run(
        f"nvidia-smi --query-gpu={query} --format={fmt}",
        shell=True, capture_output=True, text=True
    )
    return r.stdout.strip().split("\n")[0].strip()

def get_temp():
    try:
        return float(_smi("temperature.gpu", units=True))
    except Exception:
        return 99.0

def get_sm_clock():
    try:
        return float(_smi("clocks.sm", units=True))
    except Exception:
        return 0.0

def get_power():
    try:
        return float(_smi("power.draw", units=True))
    except Exception:
        return 0.0

def cooldown(label=""):
    """Wait until GPU temp < target or max wait exceeded. Return actual temps."""
    temp_start = get_temp()
    t0 = time.time()
    while True:
        temp = get_temp()
        if temp < COOLDOWN_TARGET_TEMP_C:
            break
        elapsed = time.time() - t0
        if elapsed >= COOLDOWN_MAX_WAIT_S:
            log.warning(f"[{label}] Cooldown timeout after {elapsed:.0f}s — temp {temp}°C > {COOLDOWN_TARGET_TEMP_C}°C")
            break
        time.sleep(2)
    temp_end = get_temp()
    return temp_start, temp_end

def snapshot():
    """Return current temp, SM clock, power."""
    return {
        "temp_c": get_temp(),
        "sm_clock_mhz": get_sm_clock(),
        "power_w": get_power(),
    }
