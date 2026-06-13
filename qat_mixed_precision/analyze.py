"""Aggregate runs -> results/analysis.parquet and print RQ1 (online vs posthoc).

RQ1: paired Δppl (posthoc - online) at matched budget B*, per model/seed; Wilcoxon.
Pareto table: PPL vs avg_bits per method.
"""
from __future__ import annotations
import glob
import pandas as pd
import numpy as np
from scipy import stats

import config as C


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

    print("\n=== RQ1: online vs posthoc at matched budget (paired) ===")
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


if __name__ == "__main__":
    main()
