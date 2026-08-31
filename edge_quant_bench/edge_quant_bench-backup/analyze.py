"""Aggregate raw JSON results -> results.parquet + compute realization ratio."""
import json, os
from pathlib import Path
import pandas as pd
import numpy as np

import sys
sys.path.insert(0, os.path.dirname(__file__))
import config as C


def load_raw():
    rows = []
    for p in sorted(Path(C.RAW_DIR).glob("*.json")):
        try:
            d = json.loads(p.read_text())
            # Flatten accuracy dict
            acc = d.pop("accuracy", {})
            d["accuracy_metric"] = acc.get("metric", "none")
            d["accuracy_cosine"] = acc.get("cosine_sim", None)
            d["accuracy_max_diff"] = acc.get("max_abs_diff", None)
            d["accuracy_ppl"] = acc.get("ppl", None)
            d["accuracy_flag"] = acc.get("flag", False)
            # Flatten kernel_mix to JSON string for parquet storage
            km = d.pop("kernel_mix", [])
            d["kernel_mix_json"] = json.dumps(km)
            rows.append(d)
        except Exception as e:
            print(f"Warning: could not parse {p.name}: {e}")
    return pd.DataFrame(rows)


def compute_realization_ratio(df: pd.DataFrame) -> pd.DataFrame:
    """
    R = realized_speedup / theoretical_speedup
    realized_speedup = latency(FP16, compile) / latency(scheme, backend)
    theoretical_speedup from config.THEORETICAL_SPEEDUP
    """
    # Build FP16 compile reference latencies
    fp16_ref = df[(df["scheme"] == "fp16") & (df["backend"] == "compile")][
        ["model", "batch", "seq", "latency_mean_ms"]
    ].rename(columns={"latency_mean_ms": "fp16_latency_ms"})

    df = df.merge(fp16_ref, on=["model", "batch", "seq"], how="left")

    def realized_speedup(row):
        if row["latency_mean_ms"] <= 0 or row["fp16_latency_ms"] <= 0:
            return None
        return row["fp16_latency_ms"] / row["latency_mean_ms"]

    df["realized_speedup"] = df.apply(realized_speedup, axis=1)

    # Theoretical speedup: use memory-bound model for batch<=4, compute-bound for batch>=16
    def theoretical_speedup(row):
        t = C.THEORETICAL_SPEEDUP.get(row["scheme"], None)
        if t is None:
            return None
        if row["batch"] <= 4:
            return t["memory_bound"]
        return t["compute_bound"]

    df["theoretical_speedup"] = df.apply(theoretical_speedup, axis=1)

    def realization_ratio(row):
        if row["theoretical_speedup"] and row["theoretical_speedup"] > 0 and row["realized_speedup"] is not None:
            return row["realized_speedup"] / row["theoretical_speedup"]
        return None

    df["realization_ratio"] = df.apply(realization_ratio, axis=1)

    return df


def main():
    print("Loading raw results...")
    df = load_raw()
    print(f"Loaded {len(df)} rows from {C.RAW_DIR}")

    if df.empty:
        print("No results found.")
        return df

    df = compute_realization_ratio(df)

    out = Path(C.RESULTS_DIR) / "results.parquet"
    df.to_parquet(out, index=False)
    print(f"Saved {out} ({len(df)} rows)")

    # Summary stats
    print("\n=== Realization Ratio Summary (compile backend) ===")
    compile_df = df[df["backend"] == "compile"]
    if not compile_df.empty:
        summary = compile_df.groupby(["model", "scheme"])["realization_ratio"].agg(
            ["mean", "min", "max", "count"]
        ).round(3)
        print(summary.to_string())

    print("\n=== INT8 GEMM Dispatch (W8A8, compile) ===")
    w8a8 = df[(df["scheme"] == "w8a8") & (df["backend"] == "compile")]
    if not w8a8.empty:
        print(w8a8[["model", "batch", "int8_gemm_dispatched", "realized_speedup"]].to_string())

    return df


if __name__ == "__main__":
    main()
