"""Figures from results/{analysis.parquet, runs/, audits/}. Run after analyze.py.

Writes the paper's three figures under results/figures/ (matching the basenames
used in paper_mixed_precision/.../IEEE-conference-template-062824.tex --
\\includegraphics{fig_pareto.pdf} etc.): copy them over when ready to compile.
  fig_pareto.pdf     <- pareto(model)                  (\\label{fig:pareto})
  fig_traj.pdf       <- trajectory(model, budget)       (\\label{fig:traj})
  fig_allocation.pdf <- allocation_by_projection(model) (\\label{fig:alloc})
Each also writes a plain <name>_<model>[...].png for a quick local look.
"""
from __future__ import annotations
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import config as C


def _paper_named(fig, stem: str, model: str):
    """Save the paper-matching basename only for the primary model (the one the
    figures in the .tex are actually scoped to); always save a model-suffixed
    PNG for quick inspection regardless of model."""
    if model == C.PRIMARY_MODEL:
        out_pdf = C.FIGS / f"{stem}.pdf"
        fig.savefig(out_pdf)
        print("wrote", out_pdf)


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
    ax.set_yscale("log")
    ax.set_title(f"PPL vs bits — {model}")
    ax.legend()
    fig.tight_layout()
    out = C.FIGS / f"pareto_{model}.png"
    fig.savefig(out, dpi=150)
    print("wrote", out)
    _paper_named(fig, "fig_pareto", model)
    plt.close(fig)


# ---------------------------------------------------------------------------
# fig:traj -- co-adaptation trajectory (loss + avg_bits vs step)
# ---------------------------------------------------------------------------

def trajectory(model: str, budget: int, ref_method: str = "posthoc"):
    """Online (mean +/- 1 sd band across seeds) loss and avg_bits vs step, with
    `ref_method`'s mean final loss as a dashed reference line -- matches
    fig:traj's caption ("training loss and average bit-width vs. step, with
    post-hoc final loss (dashed)")."""
    run_ids = sorted(p.stem for p in C.RUNS.glob(f"{model}__online__B{budget}__seed*.parquet"))
    if not run_ids:
        print(f"no online B{budget} runs for {model} yet -- skipping trajectory()")
        return
    dfs = [pd.read_parquet(C.RUNS / f"{rid}.parquet") for rid in run_ids]
    n = min(len(d) for d in dfs)   # in case a run logged one fewer/extra row
    steps = dfs[0]["step"].values[:n]
    loss = np.stack([d["loss"].values[:n] for d in dfs])
    bits = np.stack([d["avg_bits"].values[:n] for d in dfs])
    loss_mean, loss_sd = loss.mean(0), loss.std(0)
    bits_mean, bits_sd = bits.mean(0), bits.std(0)

    ref_final_loss = None
    ref_files = list(C.RUNS.glob(f"{model}__{ref_method}__B{budget}__seed*.parquet"))
    if ref_files:
        ref_final_loss = float(np.mean([pd.read_parquet(f)["loss"].iloc[-1] for f in ref_files]))

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(6.5, 6), sharex=True)
    ax1.plot(steps, loss_mean, "-", color="C0", label="co-adaptive (online)")
    ax1.fill_between(steps, loss_mean - loss_sd, loss_mean + loss_sd, alpha=0.2, color="C0")
    if ref_final_loss is not None:
        ax1.axhline(ref_final_loss, ls="--", color="C1", label=f"{ref_method} (final)")
    ax1.set_ylabel("training loss")
    ax1.set_title(f"Co-adaptive dynamics at B{budget} — {model} (n={len(dfs)} seeds)")
    ax1.legend()

    ax2.plot(steps, bits_mean, "-", color="C2")
    ax2.fill_between(steps, bits_mean - bits_sd, bits_mean + bits_sd, alpha=0.2, color="C2")
    ax2.set_xlabel("step")
    ax2.set_ylabel("avg weight bits")
    fig.tight_layout()

    out = C.FIGS / f"traj_{model}_B{budget}.png"
    fig.savefig(out, dpi=150)
    print("wrote", out)
    _paper_named(fig, "fig_traj", model)
    plt.close(fig)


# ---------------------------------------------------------------------------
# fig:alloc -- per-projection-type achieved bit-width, online vs post-hoc
# ---------------------------------------------------------------------------

def _final_bitmap(run_id: str) -> dict:
    """Per-layer final bit-width for one run: the last controller 'state' event
    (online) or the one-shot 'posthoc_assign' event (posthoc)."""
    p = C.AUDITS / f"{run_id}.jsonl"
    if not p.exists():
        return {}
    bitmap = {}
    with open(p) as f:
        for line in f:
            e = json.loads(line)
            if e.get("action") in ("state", "posthoc_assign") and "bits" in e:
                bitmap = e["bits"]   # keep overwriting -> ends on the last one
    return bitmap


def allocation_by_projection(model: str, budgets=(3, 4, 5, 6),
                             methods=("online", "posthoc")):
    """Mean achieved bit-width per projection-type suffix (e.g. in_proj/x_proj/
    dt_proj/out_proj for mamba-130m), averaged over seeds, grouped by budget."""
    suffixes = C.MODELS[model]["target_suffixes"]
    rows = []
    for method in methods:
        for budget in budgets:
            run_ids = sorted(p.stem for p in
                             C.RUNS.glob(f"{model}__{method}__B{budget}__seed*.parquet"))
            per_suffix = {s: [] for s in suffixes}
            for rid in run_ids:
                for name, b in _final_bitmap(rid).items():
                    suf = name.split(".")[-1]
                    if suf in per_suffix and b is not None:
                        per_suffix[suf].append(b)
            for suf, vals in per_suffix.items():
                if vals:
                    rows.append(dict(method=method, budget=budget, proj=suf,
                                     mean_bits=float(np.mean(vals))))
    if not rows:
        print(f"no per-layer audit bit data for {model} yet -- "
              f"skipping allocation_by_projection()")
        return
    df = pd.DataFrame(rows)
    present_budgets = [b for b in budgets if b in set(df.budget)]

    fig, axes = plt.subplots(1, len(present_budgets),
                             figsize=(3.2 * len(present_budgets), 3.2), sharey=True)
    axes = np.atleast_1d(axes)
    x = np.arange(len(suffixes))
    width = 0.35
    for ax, budget in zip(axes, present_budgets):
        sub = df[df.budget == budget]
        for i, method in enumerate(methods):
            vals = [sub[(sub.method == method) & (sub.proj == s)].mean_bits.mean()
                    for s in suffixes]
            ax.bar(x + (i - 0.5) * width, vals, width, label=method)
        ax.set_xticks(x)
        ax.set_xticklabels(suffixes, rotation=45, ha="right")
        ax.set_title(f"B{budget}")
    axes[0].set_ylabel("mean achieved bits")
    axes[-1].legend()
    fig.suptitle(f"Per-projection allocation — {model}")
    fig.tight_layout()

    out = C.FIGS / f"allocation_{model}.png"
    fig.savefig(out, dpi=150)
    print("wrote", out)
    _paper_named(fig, "fig_allocation", model)
    plt.close(fig)


if __name__ == "__main__":
    import sys
    model = sys.argv[1] if len(sys.argv) > 1 else C.PRIMARY_MODEL
    pareto(model)
    trajectory(model, budget=3)
    allocation_by_projection(model)
