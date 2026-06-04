"""
Generate all figures from results.parquet.
Run: python plots.py
All figures saved to results/figures/.
"""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))

import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from pathlib import Path
import config as C
from analyze import load_raw, compute_realization_ratio

FIGS = Path(C.FIGURES_DIR)
FIGS.mkdir(parents=True, exist_ok=True)

SCHEME_ORDER = ["fp32", "fp16", "bf16", "w8a16", "w4a16", "w8a8"]
SCHEME_LABELS = {"fp32": "FP32", "fp16": "FP16", "bf16": "BF16",
                 "w8a16": "W8A16", "w4a16": "W4A16", "w8a8": "W8A8"}
MODEL_ORDER = ["distilbert-base", "bert-base", "gpt2", "vit-s"]
COLORS = plt.cm.tab10.colors


def load_df():
    p = Path(C.RESULTS_DIR) / "results.parquet"
    if p.exists():
        return pd.read_parquet(p)
    df = load_raw()
    return compute_realization_ratio(df)


# ── Figure 1: Realization ratio heatmap ───────────────────────────────────────

def fig1_heatmap(df):
    """Heatmap of R over (batch × scheme), one panel per model (compile backend)."""
    comp = df[(df["backend"] == "compile") & (~df["oom"].fillna(False))]
    models = [m for m in MODEL_ORDER if m in comp["model"].unique()]
    schemes = [s for s in SCHEME_ORDER if s in comp["scheme"].unique()]

    n_models = len(models)
    fig, axes = plt.subplots(1, n_models, figsize=(5 * n_models, 4), squeeze=False)

    for ax, model in zip(axes[0], models):
        sub = comp[comp["model"] == model]
        pivot = sub.pivot_table(index="batch", columns="scheme",
                                values="realization_ratio", aggfunc="mean")
        pivot = pivot.reindex(columns=[s for s in schemes if s in pivot.columns])
        pivot.index = [f"B={b}" for b in pivot.index]
        pivot.columns = [SCHEME_LABELS.get(s, s) for s in pivot.columns]

        # Diverging colormap centered at 1.0
        vmax = max(2.0, float(pivot.max().max()) if not pivot.empty else 2.0)
        cmap = plt.cm.RdYlGn
        im = ax.imshow(pivot.values.astype(float), aspect="auto",
                       cmap=cmap, vmin=0, vmax=vmax)
        ax.set_xticks(range(len(pivot.columns)))
        ax.set_xticklabels(pivot.columns, rotation=45, ha="right", fontsize=9)
        ax.set_yticks(range(len(pivot.index)))
        ax.set_yticklabels(pivot.index, fontsize=9)
        ax.set_title(model, fontsize=11)

        for i in range(len(pivot.index)):
            for j in range(len(pivot.columns)):
                val = pivot.values[i, j]
                if not np.isnan(val):
                    ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                            fontsize=7, color="black")

        plt.colorbar(im, ax=ax, label="R (realized/theoretical)")

    fig.suptitle("Realization Ratio Heatmap (compile backend)\nGreen=quant pays off, Red=quant tax",
                 fontsize=12)
    plt.tight_layout()
    out = FIGS / "fig1_realization_heatmap.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out}")


# ── Figure 2: Latency vs batch line plots ─────────────────────────────────────

def fig2_latency_vs_batch(df):
    models = [m for m in MODEL_ORDER if m in df["model"].unique()]
    fig, axes = plt.subplots(1, len(models), figsize=(5 * len(models), 4), squeeze=False)

    for ax, model in zip(axes[0], models):
        sub = df[(df["model"] == model) & (df["backend"] == "compile") &
                 (~df["oom"].fillna(False)) & (df["latency_mean_ms"] > 0)]
        schemes = [s for s in SCHEME_ORDER if s in sub["scheme"].unique()]
        for i, scheme in enumerate(schemes):
            ss = sub[sub["scheme"] == scheme].sort_values("batch")
            if ss.empty:
                continue
            ax.semilogy(ss["batch"], ss["latency_mean_ms"], "o-",
                        color=COLORS[i % len(COLORS)],
                        label=SCHEME_LABELS.get(scheme, scheme))
            ax.fill_between(ss["batch"],
                            ss["latency_mean_ms"] - ss["latency_std_ms"],
                            ss["latency_mean_ms"] + ss["latency_std_ms"],
                            alpha=0.15, color=COLORS[i % len(COLORS)])

        ax.set_xlabel("Batch size")
        ax.set_ylabel("Latency (ms, log)")
        ax.set_title(model)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    fig.suptitle("Latency vs Batch Size (compile backend)", fontsize=12)
    plt.tight_layout()
    out = FIGS / "fig2_latency_vs_batch.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out}")


# ── Figure 3: Realization gap bars ────────────────────────────────────────────

def fig3_realization_gap(df):
    comp = df[(df["backend"] == "compile") & (~df["oom"].fillna(False))]
    # Focus on batch=16 (compute-ish regime)
    sub = comp[(comp["batch"] == 16) & comp["scheme"].isin(["w8a16", "w4a16", "w8a8"])]

    if sub.empty:
        print("fig3: no data for batch=16 quant schemes, skipping")
        return

    models = [m for m in MODEL_ORDER if m in sub["model"].unique()]
    schemes = ["w8a16", "w4a16", "w8a8"]
    x = np.arange(len(models))
    width = 0.25

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    ax1, ax2 = axes

    for i, scheme in enumerate(schemes):
        ss = sub[sub["scheme"] == scheme]
        theoretical = [ss[ss["model"] == m]["theoretical_speedup"].mean() for m in models]
        realized = [ss[ss["model"] == m]["realized_speedup"].mean() for m in models]
        offset = (i - 1) * width

        ax1.bar(x + offset, theoretical, width, alpha=0.4,
                color=COLORS[i], label=f"{SCHEME_LABELS[scheme]} theoretical")
        ax1.bar(x + offset, realized, width, alpha=0.9,
                color=COLORS[i], label=f"{SCHEME_LABELS[scheme]} realized")

        ratios = [ss[ss["model"] == m]["realization_ratio"].mean() for m in models]
        ax2.bar(x + offset, ratios, width, color=COLORS[i],
                label=SCHEME_LABELS[scheme])

    ax1.axhline(1.0, color="k", linestyle="--", linewidth=1, label="FP16 baseline")
    ax1.set_xticks(x); ax1.set_xticklabels(models, rotation=20, ha="right")
    ax1.set_ylabel("Speedup vs FP16"); ax1.set_title("Theoretical vs Realized Speedup (batch=16)")
    ax1.legend(fontsize=7)

    ax2.axhline(1.0, color="k", linestyle="--", linewidth=1, label="Perfect realization")
    ax2.set_xticks(x); ax2.set_xticklabels(models, rotation=20, ha="right")
    ax2.set_ylabel("R = realized / theoretical")
    ax2.set_title("Realization Ratio (batch=16)")
    ax2.legend(fontsize=8)

    plt.tight_layout()
    out = FIGS / "fig3_realization_gap.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out}")


# ── Figure 4: Kernel mix breakdown ────────────────────────────────────────────

def fig4_kernel_mix(df):
    """Top kernels for 3 representative cells."""
    cells = [
        {"model": "bert-base", "scheme": "w8a16", "backend": "compile", "batch": 1},
        {"model": "bert-base", "scheme": "w8a8",  "backend": "compile", "batch": 64},
        {"model": "bert-base", "scheme": "fp16",  "backend": "compile", "batch": 16},
    ]
    valid_cells = []
    for cell in cells:
        row = df[(df["model"] == cell["model"]) &
                 (df["scheme"] == cell["scheme"]) &
                 (df["backend"] == cell["backend"]) &
                 (df["batch"] == cell["batch"])]
        if not row.empty and "kernel_mix_json" in row.columns:
            km = json.loads(row.iloc[0]["kernel_mix_json"])
            if km:
                valid_cells.append((cell, km))

    if not valid_cells:
        print("fig4: no kernel mix data available, skipping")
        return

    fig, axes = plt.subplots(1, len(valid_cells), figsize=(6 * len(valid_cells), 5), squeeze=False)
    for ax, (cell, km) in zip(axes[0], valid_cells):
        top = km[:8]
        names = [k["name"][-35:] for k in top]
        times = [k["cuda_time_ms"] for k in top]
        ax.barh(range(len(names)), times, color=COLORS[:len(names)])
        ax.set_yticks(range(len(names)))
        ax.set_yticklabels(names, fontsize=7)
        ax.set_xlabel("CUDA time (ms)")
        label = f"{cell['model']}\n{cell['scheme']} | B={cell['batch']}"
        ax.set_title(label, fontsize=9)

    fig.suptitle("Kernel Mix Breakdown (top kernels by CUDA time)", fontsize=12)
    plt.tight_layout()
    out = FIGS / "fig4_kernel_mix.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out}")


# ── Figure 5: INT8 GEMM dispatch map ──────────────────────────────────────────

def fig5_int8_dispatch(df):
    """Binary heatmap: did real INT8 GEMM run? (model × scheme × batch)."""
    comp = df[df["backend"] == "compile"]
    quant_schemes = [s for s in ["w8a16", "w4a16", "w8a8"] if s in comp["scheme"].unique()]
    models = [m for m in MODEL_ORDER if m in comp["model"].unique()]

    if not quant_schemes or not models:
        print("fig5: no quant schemes data, skipping")
        return

    batches = sorted(comp["batch"].unique())
    # Build a 3D matrix: model × (scheme×batch) → bool
    fig, axes = plt.subplots(1, len(models), figsize=(4 * len(models), 4), squeeze=False)

    for ax, model in zip(axes[0], models):
        sub = comp[comp["model"] == model]
        cols = [f"{SCHEME_LABELS.get(s,s)}\nB={b}" for s in quant_schemes for b in batches]
        data = []
        for s in quant_schemes:
            row_vals = []
            for b in batches:
                r = sub[(sub["scheme"] == s) & (sub["batch"] == b)]
                if r.empty:
                    row_vals.append(np.nan)
                else:
                    row_vals.append(float(r.iloc[0].get("int8_gemm_dispatched", False)))
            data.append(row_vals)

        data_arr = np.array(data, dtype=float)
        im = ax.imshow(data_arr, aspect="auto", cmap="RdYlGn", vmin=0, vmax=1)
        ax.set_xticks(range(len(batches)))
        ax.set_xticklabels([f"B={b}" for b in batches], fontsize=8)
        ax.set_yticks(range(len(quant_schemes)))
        ax.set_yticklabels([SCHEME_LABELS.get(s, s) for s in quant_schemes], fontsize=8)
        ax.set_title(model, fontsize=10)
        plt.colorbar(im, ax=ax, label="INT8 GEMM dispatched")

    fig.suptitle("INT8 GEMM Dispatch Map (compile backend)\n"
                 "Green=real INT8 tensor-core used, Red=FP fallback (H2 test)", fontsize=11)
    plt.tight_layout()
    out = FIGS / "fig5_int8_dispatch.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out}")


# ── Figure 6: Roofline (analytical fallback) ──────────────────────────────────

def fig6_roofline(df):
    """Roofline plot using analytical arithmetic intensity."""
    # AD107 roofline parameters (approximate)
    PEAK_COMPUTE_TFLOPS_FP16 = 22.0   # RTX 4060 Laptop ~22 TFLOPS FP16
    PEAK_BW_TB_S = 0.272              # 272 GB/s memory bandwidth

    ridge_point = PEAK_COMPUTE_TFLOPS_FP16 * 1e12 / (PEAK_BW_TB_S * 1e12)  # FLOPs/byte

    # Build AI from analytical model
    comp = df[(df["backend"] == "compile") & (df["flops"] > 0) &
              (df["analytical_mem_bytes"] > 0) & (~df["oom"].fillna(False))]
    if comp.empty:
        print("fig6: no data with FLOPs, skipping")
        return

    comp = comp.copy()
    comp["arith_intensity"] = comp["flops"] / comp["analytical_mem_bytes"]
    comp["compute_throughput_tflops"] = (
        comp["flops"] / (comp["latency_mean_ms"] / 1000.0) / 1e12
    ).where(comp["latency_mean_ms"] > 0)

    fig, ax = plt.subplots(figsize=(9, 6))

    # Draw roofline
    ai_range = np.logspace(-2, 3, 500)
    roofline = np.minimum(ai_range * PEAK_BW_TB_S * 1e12,
                          PEAK_COMPUTE_TFLOPS_FP16 * 1e12) / 1e12
    ax.loglog(ai_range, roofline, "k-", linewidth=2, label="AD107 FP16 Roofline (analytical)")
    ax.axvline(ridge_point, color="k", linestyle=":", alpha=0.5)

    markers = {"distilbert-base": "o", "bert-base": "s", "gpt2": "^", "vit-s": "D"}
    for i, scheme in enumerate(SCHEME_ORDER):
        sub = comp[comp["scheme"] == scheme]
        if sub.empty:
            continue
        for model, marker in markers.items():
            ms = sub[sub["model"] == model]
            if ms.empty:
                continue
            ax.scatter(ms["arith_intensity"], ms["compute_throughput_tflops"],
                       marker=marker, color=COLORS[i % len(COLORS)], s=60, alpha=0.8,
                       label=f"{SCHEME_LABELS.get(scheme, scheme)}/{model}" if model == "bert-base" else "")

    ax.set_xlabel("Arithmetic Intensity (FLOPs/byte) — ANALYTICAL MODEL")
    ax.set_ylabel("Performance (TFLOPS)")
    ax.set_title("Roofline Plot — AD107 (RTX 4060 Laptop)\n"
                 "NOTE: Arithmetic intensity from analytical model, not Nsight Compute")
    ax.legend(fontsize=7, loc="lower right")
    ax.grid(True, which="both", alpha=0.3)

    plt.tight_layout()
    out = FIGS / "fig6_roofline_analytical.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out}")


# ── Figure 7: Decision table ───────────────────────────────────────────────────

def fig7_decision_table(df):
    """Best scheme per (model-kind, batch regime) based on realized speedup."""
    comp = df[(df["backend"] == "compile") & (~df["oom"].fillna(False)) &
              (df["latency_mean_ms"] > 0)]

    if comp.empty:
        print("fig7: no data, skipping")
        return

    kind_map = {"distilbert-base": "encoder", "bert-base": "encoder",
                "gpt2": "decoder", "vit-s": "vit", "resnet50": "cnn"}
    comp = comp.copy()
    comp["kind"] = comp["model"].map(kind_map).fillna("unknown")
    comp["batch_regime"] = comp["batch"].apply(lambda b: "small (B≤4)" if b <= 4 else "large (B≥16)")

    pivot = comp.groupby(["kind", "batch_regime", "scheme"])["realized_speedup"].mean()

    rows = []
    for (kind, regime), group in pivot.groupby(level=[0, 1]):
        group = group.dropna()
        if group.empty:
            continue
        best_scheme = group.idxmax()[-1]
        best_val = group.max()
        rows.append({
            "Model Kind": kind,
            "Batch Regime": regime,
            "Best Scheme": SCHEME_LABELS.get(best_scheme, best_scheme),
            "Realized Speedup vs FP16": round(best_val, 3),
        })

    table_df = pd.DataFrame(rows).sort_values(["Model Kind", "Batch Regime"])

    fig, ax = plt.subplots(figsize=(9, max(3, len(rows) * 0.5 + 1)))
    ax.axis("off")
    t = ax.table(
        cellText=table_df.values,
        colLabels=table_df.columns,
        cellLoc="center", loc="center",
    )
    t.auto_set_font_size(False)
    t.set_fontsize(10)
    t.scale(1, 1.5)
    ax.set_title("Decision Table: Best Quantization Scheme\n(compile backend, FP16 baseline)",
                 fontsize=12, pad=20)

    plt.tight_layout()
    out = FIGS / "fig7_decision_table.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out}")

    # Also save as CSV
    table_df.to_csv(FIGS / "decision_table.csv", index=False)
    print(f"Saved {FIGS / 'decision_table.csv'}")


# ── Figure 8: Thermal appendix ────────────────────────────────────────────────

def fig8_thermal(df):
    """GPU temp and SM clock per config."""
    if "temp_start" not in df.columns or df["temp_start"].isna().all():
        print("fig8: no thermal data, skipping")
        return

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    df_sorted = df.sort_values("timestamp") if "timestamp" in df.columns else df
    x = range(len(df_sorted))

    axes[0].plot(x, df_sorted["temp_start"].fillna(0), "o", markersize=2, alpha=0.6, label="temp_start")
    axes[0].axhline(C.COOLDOWN_TARGET_TEMP_C, color="r", linestyle="--", label=f"Cooldown target ({C.COOLDOWN_TARGET_TEMP_C}°C)")
    axes[0].set_xlabel("Config index"); axes[0].set_ylabel("GPU Temp (°C)")
    axes[0].set_title("GPU Temperature Per Config"); axes[0].legend()

    axes[1].plot(x, df_sorted["sm_clock_mhz"].fillna(0), "o", markersize=2, alpha=0.6, color="orange")
    axes[1].set_xlabel("Config index"); axes[1].set_ylabel("SM Clock (MHz)")
    axes[1].set_title("SM Clock Per Config (no lock — sudo unavailable)")

    plt.tight_layout()
    out = FIGS / "fig8_thermal_appendix.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out}")


def main():
    print("Loading results...")
    df = load_df()
    if df.empty:
        print("No results to plot.")
        return

    print(f"Plotting {len(df)} rows...")
    fig1_heatmap(df)
    fig2_latency_vs_batch(df)
    fig3_realization_gap(df)
    fig4_kernel_mix(df)
    fig5_int8_dispatch(df)
    fig6_roofline(df)
    fig7_decision_table(df)
    fig8_thermal(df)
    print("Done. All figures in results/figures/")


if __name__ == "__main__":
    main()
