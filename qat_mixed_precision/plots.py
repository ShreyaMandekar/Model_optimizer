"""Figures from results/analysis.parquet. Run after analyze.py."""
from __future__ import annotations
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import config as C


def pareto(model: str):
    df = pd.read_parquet(C.RESULTS / "analysis.parquet")
    df = df[df.model == model]
    fig, ax = plt.subplots(figsize=(5, 4))
    for method, mk in [("uniform", "s--"), ("posthoc", "o-"), ("online", "^-")]:
        sub = (df[df.method == method]
               .groupby("tag").agg(ppl=("final_ppl", "mean"),
                                   bits=("final_avg_bits", "mean"))
               .sort_values("bits"))
        if len(sub):
            ax.plot(sub.bits, sub.ppl, mk, label=method)
    ax.set_xlabel("average weight bits")
    ax.set_ylabel("perplexity (WikiText-2)")
    ax.set_title(f"PPL vs bits — {model}")
    ax.legend()
    fig.tight_layout()
    out = C.FIGS / f"pareto_{model}.png"
    fig.savefig(out, dpi=150)
    print("wrote", out)


if __name__ == "__main__":
    import sys
    pareto(sys.argv[1] if len(sys.argv) > 1 else C.PRIMARY_MODEL)
