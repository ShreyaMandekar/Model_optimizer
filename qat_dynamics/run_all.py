"""Crash-safe sweep over the 2×4 grid × seeds × models.

Skips already-completed runs (one parquet per run).
Usage:
    python run_all.py [--models distilbert bert] [--tasks sst2] [--phase 1]
"""

import argparse
import traceback
from config import ARMS, SEEDS, RUNS

from train_qat import train_run

PRIMARY_MODELS = ["distilbert"]
PHASE2_MODELS  = ["bert"]
DEFAULT_TASK   = "sst2"


def sweep(models, task, seeds):
    total = len(models) * len(ARMS) * len(seeds)
    done  = 0
    failed = []

    for model_key in models:
        for metric, strategy in ARMS:
            for seed in seeds:
                run_id = f"{model_key}__{task}__{metric}__{strategy}__seed{seed}"
                out = RUNS / f"{run_id}.parquet"
                if out.exists():
                    print(f"[skip] {run_id}")
                    done += 1
                    continue
                print(f"\n[{done+1}/{total}] {run_id}")
                try:
                    result = train_run(model_key, task, metric, strategy, seed)
                    done += 1
                    print(f"  acc={result.get('final_acc', '?'):.4f} "
                          f"frozen={result.get('frozen_frac', '?'):.2f}")
                except Exception as e:
                    print(f"  FAILED: {e}")
                    traceback.print_exc()
                    failed.append(run_id)

    print(f"\n{'='*60}")
    print(f"Sweep done: {done}/{total} completed, {len(failed)} failed")
    if failed:
        print("Failed runs:")
        for r in failed:
            print(f"  {r}")
    return failed


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", nargs="+", default=PRIMARY_MODELS)
    parser.add_argument("--task",   default=DEFAULT_TASK)
    parser.add_argument("--phase",  type=int, default=1,
                        help="1=primary models, 2=all models")
    args = parser.parse_args()

    if args.phase == 2:
        models = PRIMARY_MODELS + PHASE2_MODELS
    else:
        models = args.models

    sweep(models, args.task, SEEDS)


if __name__ == "__main__":
    main()
