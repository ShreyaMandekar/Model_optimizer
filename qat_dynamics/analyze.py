"""Analysis: compute RQ1-RQ5 statistics from run parquets → analysis.parquet.

Outputs:
  results/analysis.parquet    — joined table for plots
  Prints correlation tables, ablation grid, main effects.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import pyarrow as pa
from scipy import stats

from config import RUNS, RESULTS, METRICS, STRATEGIES


# ---------------------------------------------------------------------------
# Load all runs
# ---------------------------------------------------------------------------

def load_runs() -> pd.DataFrame:
    rows = []
    for f in sorted(RUNS.glob("*.parquet")):
        tbl = pq.read_table(f)
        df = tbl.to_pandas()
        # last row is the summary row
        summary = df.iloc[-1]
        row = {
            "run_id":    str(summary.get("run_id", f.stem)),
            "model":     str(summary.get("model", "")),
            "task":      str(summary.get("task", "")),
            "metric":    str(summary.get("metric", "")),
            "strategy":  str(summary.get("strategy", "")),
            "seed":      int(summary.get("seed", 0)),
            "final_acc": float(summary.get("final_acc", float("nan"))),
            "final_loss":float(summary.get("final_loss", float("nan"))),
            "frozen_frac":float(summary.get("frozen_frac", float("nan"))),
            "n_layers":  int(summary.get("n_layers", 0)),
            "n_frozen":  int(summary.get("n_frozen", 0)),
        }
        rows.append(row)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# RQ1: CV vs KL correlation (requires per-layer data from a probe run)
# ---------------------------------------------------------------------------

def compute_rq1(df: pd.DataFrame):
    """Extract CV and KL per layer from one probe run and compute correlation."""
    # find a run with both cv and kl data logged per layer
    # we look for kl-metric runs to get KL values, and cv-metric runs for CV
    kl_runs  = df[(df["metric"] == "kl") & (df["strategy"] == "s1_oneway")]
    cv_runs  = df[(df["metric"] == "cv") & (df["strategy"] == "s1_oneway")]
    if kl_runs.empty or cv_runs.empty:
        print("[RQ1] Insufficient data for correlation.")
        return {}

    # load trajectory parquet from first kl and cv run (same model/task/seed)
    kl_row = kl_runs.iloc[0]
    cv_row = cv_runs[(cv_runs["model"] == kl_row["model"]) &
                     (cv_runs["seed"]  == kl_row["seed"])].iloc[0] \
             if not cv_runs[(cv_runs["model"] == kl_row["model"]) &
                            (cv_runs["seed"]  == kl_row["seed"])].empty else cv_runs.iloc[0]

    kl_path = RUNS / f"{kl_row['run_id']}.parquet"
    cv_path = RUNS / f"{cv_row['run_id']}.parquet"
    kl_traj = pq.read_table(kl_path).to_pandas()
    cv_traj = pq.read_table(cv_path).to_pandas()

    # extract final per-layer CV and KL from last logged step
    cv_cols  = [c for c in cv_traj.columns if c.startswith("cv__")]
    kl_audit_path = Path(str(kl_path).replace("runs/", "audits/").replace(".parquet", "_audits.jsonl"))

    cv_final = cv_traj[cv_cols].dropna().iloc[-1] if len(cv_traj) > 1 else pd.Series()
    cv_dict  = {c.replace("cv__", ""): v for c, v in cv_final.items()}

    kl_dict = {}
    if kl_audit_path.exists():
        audit_rows = [json.loads(l) for l in kl_audit_path.read_text().splitlines()]
        if audit_rows:
            last_step = max(r["step"] for r in audit_rows)
            for r in audit_rows:
                if r["step"] == last_step:
                    kl_dict[r["layer"]] = r["kl"]

    common = sorted(set(cv_dict) & set(kl_dict))
    if len(common) < 3:
        print(f"[RQ1] Only {len(common)} common layers — too few for correlation.")
        return {}

    cv_vals = np.array([cv_dict[n] for n in common])
    kl_vals = np.array([kl_dict[n] for n in common])

    tau, p_tau = stats.kendalltau(cv_vals, kl_vals)
    rho, p_rho = stats.spearmanr(cv_vals, kl_vals)

    result = {"kendall_tau": tau, "p_tau": p_tau, "spearman_rho": rho, "p_rho": p_rho,
              "n_layers": len(common), "cv_vals": cv_vals.tolist(), "kl_vals": kl_vals.tolist(),
              "layer_names": common}
    print(f"\n[RQ1] CV vs KL correlation (n={len(common)} layers)")
    print(f"  Kendall τ = {tau:.3f}  (p={p_tau:.3f})")
    print(f"  Spearman ρ = {rho:.3f}  (p={p_rho:.3f})")
    h1_holds = abs(tau) < 0.5
    print(f"  H1 (|τ|<0.5, CV≠KL): {'SUPPORTED' if h1_holds else 'NOT SUPPORTED'}")
    return result


# ---------------------------------------------------------------------------
# RQ3: 2×4 ablation grid
# ---------------------------------------------------------------------------

def compute_rq3(df: pd.DataFrame):
    grp = df.groupby(["model", "metric", "strategy"])["final_acc"]
    agg = grp.agg(["mean", "std", "count"]).reset_index()
    agg.columns = ["model", "metric", "strategy", "acc_mean", "acc_std", "n_seeds"]

    print("\n[RQ3] 2×4 Ablation Grid (DistilBERT/SST-2)")
    print(f"{'strategy':<16} {'cv_mean':>9} {'cv_std':>8} {'kl_mean':>9} {'kl_std':>8}")
    print("-" * 55)
    for strat in STRATEGIES:
        cv_row = agg[(agg["metric"] == "cv") & (agg["strategy"] == strat)]
        kl_row = agg[(agg["metric"] == "kl") & (agg["strategy"] == strat)]
        cv_m = cv_row["acc_mean"].values[0] if len(cv_row) else float("nan")
        cv_s = cv_row["acc_std"].values[0]  if len(cv_row) else float("nan")
        kl_m = kl_row["acc_mean"].values[0] if len(kl_row) else float("nan")
        kl_s = kl_row["acc_std"].values[0]  if len(kl_row) else float("nan")
        print(f"{strat:<16} {cv_m:>9.4f} {cv_s:>8.4f} {kl_m:>9.4f} {kl_s:>8.4f}")

    # main effects: metric and strategy
    metric_effect = _metric_main_effect(df)
    strat_effect  = _strategy_main_effect(df)

    print(f"\n  Metric main effect (KL-CV, paired Wilcoxon): {metric_effect}")
    print(f"  Strategy main effect (best-worst row, paired Wilcoxon): {strat_effect}")
    return agg


def _metric_main_effect(df):
    cv_acc = df[df["metric"] == "cv"].groupby(["strategy", "seed"])["final_acc"].mean()
    kl_acc = df[df["metric"] == "kl"].groupby(["strategy", "seed"])["final_acc"].mean()
    shared = cv_acc.index.intersection(kl_acc.index)
    if len(shared) < 3:
        return "insufficient data"
    diffs = kl_acc[shared].values - cv_acc[shared].values
    stat, p = stats.wilcoxon(diffs) if len(diffs) >= 6 else (float("nan"), float("nan"))
    return f"KL-CV diff mean={np.mean(diffs):.4f}  Wilcoxon p={p:.3f}"


def _strategy_main_effect(df):
    best  = "s3_reactive"
    worst = "s1_oneway"
    b = df[df["strategy"] == best]["final_acc"].values
    w = df[df["strategy"] == worst]["final_acc"].values
    if min(len(b), len(w)) < 3:
        return "insufficient data"
    stat, p = stats.mannwhitneyu(b, w, alternative="greater")
    return f"{best} vs {worst} mean diff={np.mean(b)-np.mean(w):.4f}  MWU p={p:.3f}"


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    df = load_runs()
    if df.empty:
        print("No completed runs found in results/runs/. Run run_all.py first.")
        return

    print(f"Loaded {len(df)} runs")
    print(df[["model", "metric", "strategy", "seed", "final_acc", "frozen_frac"]].to_string())

    rq1 = compute_rq1(df)
    compute_rq3(df)

    # save
    out = RESULTS / "analysis.parquet"
    pq.write_table(pa.Table.from_pandas(df), out)
    print(f"\nSaved analysis to {out}")


if __name__ == "__main__":
    main()
