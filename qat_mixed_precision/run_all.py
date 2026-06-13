"""Crash-safe sweep over config.build_runs(), dispatched across N GPUs.

Each run is a subprocess (`train_qat.py --run_id ...`) pinned to one GPU via
CUDA_VISIBLE_DEVICES. Skip-if-done (parquet exists). posthoc_base runs first so
posthoc runs can reuse the shared 8-bit checkpoint.

Usage:
  python run_all.py                # use all visible GPUs
  python run_all.py --gpus 0,1     # explicit
  python run_all.py --dry          # print the plan only
"""
from __future__ import annotations
import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

import config as C


def order_runs(runs):
    # posthoc_base first (enables checkpoint sharing), then the rest
    base = [r for r in runs if r["method"] == "posthoc_base"]
    rest = [r for r in runs if r["method"] != "posthoc_base"]
    return base + rest


def pending(runs):
    return [r for r in runs if not (C.RUNS / f"{r['run_id']}.parquet").exists()]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpus", default=None, help="comma list, e.g. 0,1")
    ap.add_argument("--dry", action="store_true")
    args = ap.parse_args()

    if args.gpus:
        gpus = [g.strip() for g in args.gpus.split(",")]
    else:
        import torch
        n = torch.cuda.device_count() or 1
        gpus = [str(i) for i in range(n)]

    runs = order_runs(C.build_runs())
    todo = pending(runs)
    print(f"{len(runs)} runs total, {len(todo)} pending, GPUs={gpus}")
    if args.dry:
        for r in runs:
            mark = "done" if r not in todo else "TODO"
            print(f"  [{mark}] {r['run_id']}")
        return

    # ensure posthoc_base finishes before posthoc on the same (model, seed):
    # run all bases first (across GPUs), then the rest.
    bases = [r for r in todo if r["method"] == "posthoc_base"]
    rest  = [r for r in todo if r["method"] != "posthoc_base"]

    for phase_name, phase in [("posthoc_base", bases), ("main", rest)]:
        if not phase:
            continue
        print(f"=== phase: {phase_name} ({len(phase)} runs) ===")
        _dispatch(phase, gpus)


def _dispatch(jobs, gpus):
    procs = {}  # gpu -> (Popen, run_id)
    queue = list(jobs)
    py = sys.executable
    here = Path(__file__).parent
    while queue or procs:
        # launch on free GPUs
        for g in gpus:
            if g not in procs and queue:
                r = queue.pop(0)
                env = dict(os.environ, CUDA_VISIBLE_DEVICES=g)
                out_file = C.CONSOLE / f"{r['run_id']}.txt"
                fh = open(out_file, "w")
                p = subprocess.Popen(
                    [py, str(here / "train_qat.py"), "--run_id", r["run_id"]],
                    env=env, stdout=fh, stderr=subprocess.STDOUT)
                procs[g] = (p, r["run_id"], fh)
                print(f"  launch GPU{g}: {r['run_id']}")
        # reap finished
        time.sleep(3)
        for g in list(procs):
            p, rid, fh = procs[g]
            if p.poll() is not None:
                fh.close()
                status = "ok" if p.returncode == 0 else f"FAIL({p.returncode})"
                print(f"  GPU{g} finished {rid}: {status}")
                del procs[g]


if __name__ == "__main__":
    main()
