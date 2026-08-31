"""Generate all figures from results/analysis.parquet.

Usage:
    python plots.py
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats

from config import RESULTS, RUNS, AUDITS, FIGS, STRATEGIES, METRICS
from analyze import load_runs, compute_rq1


STRATEGY_LABELS = {
    "s1_oneway":   "S1: One-way",
    "s2_recal":    "S2: +Recal",
    "s3_reactive": "S3: Reactive",
    "s4_soft":     "S4: Soft",
}
METRIC_COLORS = {"cv": "#2196F3", "kl": "#F44336"}


# ---------------------------------------------------------------------------
# Fig 1: CV vs KL scatter (RQ1)
# ---------------------------------------------------------------------------

def fig1_cv_vs_kl(rq1: dict):
    if not rq1 or "cv_vals" not in rq1:
        print("[fig1] No RQ1 data — skipping")
        return
    cv  = np.array(rq1["cv_vals"])
    kl  = np.array(rq1["kl_vals"])
    tau = rq1["kendall_tau"]
    p   = rq1["p_tau"]

    fig, ax = plt.subplots(figsize=(5, 4))
    ax.scatter(cv, kl, alpha=0.7, edgecolors="k", linewidths=0.5)
    ax.set_xlabel("CV (%) — scale stability")
    ax.set_ylabel("KL_s→t (nats) — output sensitivity")
    ax.set_title(f"RQ1: CV vs KL  (Kendall τ={tau:.2f}, p={p:.3f})")
    ax.annotate(f"τ={tau:.2f}", xy=(0.05, 0.92), xycoords="axes fraction", fontsize=10)
    fig.tight_layout()
    out = FIGS / "fig1_cv_vs_kl.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"[fig1] saved {out}")


# ---------------------------------------------------------------------------
# Fig 3: Ablation grid (RQ3) — the headline
# ---------------------------------------------------------------------------

def fig3_ablation_grid(df: pd.DataFrame):
    grp = df.groupby(["metric", "strategy"])["final_acc"]
    agg = grp.agg(["mean", "std"]).reset_index()

    fig, ax = plt.subplots(figsize=(7, 4))
    x   = np.arange(len(STRATEGIES))
    w   = 0.35
    for i, metric in enumerate(METRICS):
        m_data = agg[agg["metric"] == metric]
        means = []
        stds  = []
        for s in STRATEGIES:
            row = m_data[m_data["strategy"] == s]
            means.append(row["mean"].values[0] if len(row) else float("nan"))
            stds.append(row["std"].values[0]   if len(row) else 0.0)
        offset = (i - 0.5) * w
        bars = ax.bar(x + offset, means, w, label=f"Metric: {metric.upper()}",
                      color=METRIC_COLORS[metric], alpha=0.85,
                      yerr=stds, capsize=3, error_kw={"linewidth": 1.2})

    ax.set_xticks(x)
    ax.set_xticklabels([STRATEGY_LABELS[s] for s in STRATEGIES], fontsize=9)
    ax.set_ylabel("Val Accuracy (mean ± std, n=3 seeds)")
    ax.set_title("RQ3: 2×4 Ablation at Matched Frozen Fraction")
    ax.legend()
    ax.set_ylim(bottom=max(0, ax.get_ylim()[0] - 0.01))
    fig.tight_layout()
    out = FIGS / "fig3_ablation_grid.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"[fig3] saved {out}")


# ---------------------------------------------------------------------------
# Fig 4: Trajectories for one run (per-layer CV/scale/state)
# ---------------------------------------------------------------------------

def fig4_trajectories(df: pd.DataFrame):
    if df.empty:
        return
    row  = df[(df["metric"] == "kl") & (df["strategy"] == "s3_reactive")].iloc[0]
    path = RUNS / f"{row['run_id']}.parquet"
    if not path.exists():
        return
    traj = pq.read_table(path).to_pandas()
    cv_cols = [c for c in traj.columns if c.startswith("cv__")]
    if not cv_cols or "step" not in traj.columns:
        return

    sub = traj.dropna(subset=["step"])
    sub = sub[sub["step"].apply(lambda x: str(x).isdigit() if isinstance(x, str) else True)]
    sub = sub[pd.to_numeric(sub["step"], errors="coerce").notna()].copy()
    sub["step"] = sub["step"].astype(float)
    sub = sub.sort_values("step").head(200)

    fig, ax = plt.subplots(figsize=(8, 4))
    for col in cv_cols[:6]:
        vals = pd.to_numeric(sub[col], errors="coerce")
        ax.plot(sub["step"], vals, alpha=0.6, linewidth=0.8, label=col.replace("cv__", "")[-30:])
    ax.set_xlabel("Training step")
    ax.set_ylabel("CV (%)")
    ax.set_title("Fig 4: Per-layer CV trajectories (KL / s3_reactive)")
    ax.legend(fontsize=6, loc="upper right", ncol=2)
    fig.tight_layout()
    out = FIGS / "fig4_trajectories.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"[fig4] saved {out}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    df   = load_runs()
    rq1  = compute_rq1(df)

    fig1_cv_vs_kl(rq1)
    if not df.empty:
        fig3_ablation_grid(df)
        fig4_trajectories(df)
    else:
        print("[plots] No run data found yet.")

    print("Done.")


if __name__ == "__main__":
    main()
