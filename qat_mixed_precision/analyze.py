"""Aggregate runs -> results/analysis.parquet and print RQ1 (online vs posthoc).

RQ1: paired Δppl (posthoc - online) at matched budget B*, per model/seed; Wilcoxon.
Pareto table: PPL vs avg_bits per method.
Paper stat: paired Δppl restricted to the sub-4-bit budgets (B3, B4) only, with a
bootstrap 95% CI -- this is the specific number reported in the paper's Results
section (mean 352.8, CI [79, 635], Wilcoxon p=0.156 as of the mamba-130m data in
results/); the broader RQ1 print above includes B5/B6 too and will not match it.
"""
from __future__ import annotations
import glob
import pandas as pd
import numpy as np
from scipy import stats

import config as C

SUB4BIT_BUDGETS = (3, 4)   # the regime the paper's headline paired stat covers
BOOTSTRAP_N = 10_000
BOOTSTRAP_SEED = 0


def collect():
    rows = []
    for p in glob.glob(str(C.RUNS / "*.parquet")):
        df = pd.read_parquet(p)
        last = df.iloc[-1]
        rows.append(dict(
            run_id=last["run_id"], model=last["model"], method=last["method"],
            tag=last["tag"], seed=int(last["seed"]),
            budget=float(last["budget"]) if pd.notna(last["budget"]) else None,
            final_ppl=float(last["final_ppl"]),
            final_avg_bits=float(last["final_avg_bits"])))
    return pd.DataFrame(rows)


def bootstrap_ci(diffs, n_resamples=BOOTSTRAP_N, ci=95, seed=BOOTSTRAP_SEED):
    """Percentile bootstrap CI on the mean of `diffs` (paired-difference resampling)."""
    diffs = np.asarray(diffs, dtype=float)
    rng = np.random.default_rng(seed)
    n = len(diffs)
    means = np.array([rng.choice(diffs, size=n, replace=True).mean()
                      for _ in range(n_resamples)])
    lo, hi = np.percentile(means, [(100 - ci) / 2, 100 - (100 - ci) / 2])
    return float(lo), float(hi)


def sub4bit_paired_stat(df, model=C.PRIMARY_MODEL, budgets=SUB4BIT_BUDGETS):
    """Reproduces the paper's headline paired-difference stat: online vs posthoc,
    restricted to the sub-4-bit budgets, bootstrap CI + Wilcoxon on the same pairs.
    """
    sub = df[(df.model == model) & (df.budget.isin(budgets))]
    on = sub[sub.method == "online"].set_index(["budget", "seed"]).final_ppl
    ph = sub[sub.method == "posthoc"].set_index(["budget", "seed"]).final_ppl
    common = on.index.intersection(ph.index)
    print(f"\n=== Paper stat: sub-4-bit (B{{{','.join(str(b) for b in budgets)}}}) "
          f"paired Δppl, {model} ===")
    if not len(common):
        print("no sub-4-bit matched pairs yet")
        return
    d = (ph[common] - on[common]).values
    lo, hi = bootstrap_ci(d)
    print(f"n={len(d)} pairs, mean Δppl (posthoc - online) = {d.mean():+.1f} "
          f"(>0 favours online)")
    print(f"bootstrap 95% CI ({BOOTSTRAP_N:,} resamples, seed={BOOTSTRAP_SEED}) "
          f"= [{lo:.0f}, {hi:.0f}]")
    if len(common) >= 6:
        w = stats.wilcoxon(ph[common], on[common])
        print(f"Wilcoxon signed-rank p = {w.pvalue:.3f} (same {len(common)} pairs)")


def main():
    df = collect()
    if df.empty:
        print("no runs yet"); return
    df.to_parquet(C.RESULTS / "analysis.parquet")
    print("=== Pareto: mean PPL by method x budget (avg_bits) ===")
    g = (df.groupby(["model", "method", "tag"])
           .agg(ppl=("final_ppl", "mean"), ppl_sd=("final_ppl", "std"),
                bits=("final_avg_bits", "mean"), n=("final_ppl", "size"))
           .reset_index().sort_values(["model", "bits"]))
    print(g.to_string(index=False))

    print("\n=== RQ1: online vs posthoc at matched budget, ALL budgets (paired) ===")
    on = df[df.method == "online"].set_index(["model", "budget", "seed"]).final_ppl
    ph = df[df.method == "posthoc"].set_index(["model", "budget", "seed"]).final_ppl
    common = on.index.intersection(ph.index)
    if len(common):
        d = (ph[common] - on[common])    # positive => online better (lower ppl)
        print(f"matched pairs: {len(common)}")
        print(f"mean Δppl (posthoc - online) = {d.mean():+.3f}  (>0 favours online)")
        if len(common) >= 6:
            w = stats.wilcoxon(ph[common], on[common])
            print(f"Wilcoxon p = {w.pvalue:.4f}")
        for B in sorted(set(i[1] for i in common)):
            sub = [d[i] for i in common if i[1] == B]
            print(f"  B*={B}: mean Δ={np.mean(sub):+.3f} over {len(sub)} pairs")
    else:
        print("no matched online/posthoc pairs yet")

    sub4bit_paired_stat(df)


if __name__ == "__main__":
    main()
