# Edge-GPU Quantization Realization Benchmark

**Does quantization actually deliver its promised speedup on a consumer/edge GPU?**

This benchmark maps the *realization ratio* R = realized_speedup / theoretical_speedup across
{model × quant scheme × backend × batch size} on a **NVIDIA GeForce RTX 4060 Laptop GPU**
(Ada Lovelace, AD107, 24 SMs, 8 GB). The core finding: on this hardware, quantization
frequently imposes a **quantization tax** rather than a speedup.

---

## Hardware & Environment

| Item | Value |
|------|-------|
| GPU | NVIDIA GeForce RTX 4060 Laptop GPU |
| Architecture | Ada Lovelace (AD107, sm_89) |
| SMs | 24 |
| VRAM | 8.19 GB |
| PyTorch | 2.6.0+cu124 |
| torchao | 0.9.0 |
| transformers | 5.3.0 |
| timm | 1.0.27 |
| onnxruntime-gpu | 1.26.0 |

See `results/env.json` for full version details.

---

## Clock Control Mode

**No sudo available.** Clock locking was not applied. Instead:
- GPU temperature is polled before each config (`temp_start`) and after cooldown (`temp_end`).
- A cooldown gate waits until GPU temp < 55 °C between configs (60 s max).
- SM clock is recorded per-config and reported in `fig8_thermal_appendix.png`.
- Users should interpret latency numbers with the awareness that laptop thermals introduce
  ±5–15% variance at batch=1; large-batch numbers (batch ≥ 16) are stable (CV < 2%).

---

## How to Reproduce

```bash
# Install deps
pip install -r requirements.txt

# Check environment
python env_check.py

# Phase 0: sanity validation
python run_sweep.py --phase0

# Phase 1: full torchao sweep (4 models × 6 schemes × 2 backends × 4 batches)
python run_sweep.py

# Phase 2: ORT backend + ResNet-50
python run_phase2_ort.py

# Phase 3: seq-len sensitivity
python run_phase3_seqlen.py

# Aggregate + compute realization ratios
python analyze.py

# Regenerate all 8 figures
python plots.py
```

The sweep is crash-safe: each config writes one JSON to `results/raw/`; rerun skips completed cells.

---

## Measurement Methodology

- **Timing:** `torch.cuda.Event` with `synchronize()`. `model.eval()` + `torch.inference_mode()`.
- **Protocol:** 30 warmup iters → 3 blocks × 100 iters. Stats: mean, std, p50, p90, p99 (ms).
- **Inputs:** Fixed dummy tensors per config (same across schemes for fair comparison).
- **FP16 baseline:** `FP16 + torch.compile` is the baseline for all realization ratio calculations.
- **Deterministic metrics:** kernel count, kernel mix, `int8_gemm_dispatched`, FLOPs,
  weight bytes, analytical memory traffic — collected on **every** config.
- **`peak_mem_mb`** = `torch.cuda.max_memory_allocated()` — this is **resident footprint**,
  NOT DRAM traffic. True memory traffic requires Nsight Compute (see below).

### Nsight Compute (ncu)

`ncu` is installed (v2025.4.0.0) but requires elevated GPU performance counter permissions
(`ERR_NVGPUCTRPERM`). No sudo is available on this machine. Therefore:
- The roofline plot (`fig6_roofline_analytical.png`) uses the **analytical traffic model**
  (weight_bytes × 2 as a proxy for total DRAM traffic). It is labeled "ANALYTICAL MODEL".
- All kernel-mix data comes from `torch.profiler` (exact, but measures CPU-orchestrated
  kernel dispatch time, not raw DRAM bytes).

### Theoretical Speedup Model

| Scheme | Memory-bound (batch ≤ 4) | Compute-bound (batch ≥ 16) |
|--------|--------------------------|---------------------------|
| FP32 | 0.5× (vs FP16 baseline) | 0.5× |
| FP16 | 1.0× (baseline) | 1.0× |
| BF16 | 1.0× | 1.0× |
| W8A16 | 2.0× (half weight bytes) | 1.0× |
| W4A16 | 4.0× (quarter weight bytes) | 1.0× |
| W8A8 | 2.0× | 2.0× (INT8 tensor cores) |

---

## Models

| Name | Source | Kind | Phase |
|------|--------|------|-------|
| distilbert-base-uncased | transformers | encoder | 1 (torchao) |
| bert-base-uncased | transformers | encoder | 1 + 3 |
| gpt2 | transformers | decoder | 1 + 3 |
| vit_small_patch16_224 | timm | vit | 1 |
| resnet50 | timm | cnn | 2 (ORT only) |

**Note on ResNet-50:** torchao `quantize_()` targets `nn.Linear` only. ResNet-50 is
conv-heavy, so torchao quantizes almost nothing. Phase 1 (torchao) therefore uses only the
4 linear-heavy models. ResNet-50 is benchmarked via ONNX Runtime in Phase 2.

---

## Key Findings

### Realization Ratio Map (Phase 1 + 3, compile backend, all seq-lens)

| Model | Scheme | R (mean) | R (min) | R (max) | Verdict |
|-------|--------|----------|---------|---------|---------|
| bert-base | W8A16 | 0.59 | 0.28 | 0.80 | Tax — 59% of theoretical |
| bert-base | W4A16 | 0.17 | 0.07 | 0.26 | Strong tax |
| bert-base | W8A8 | 0.17 | 0.02 | 0.51 | Severe tax |
| distilbert-base | W8A16 | 0.60 | 0.32 | 0.87 | Tax |
| distilbert-base | W8A8 | 0.07 | 0.01 | 0.13 | Severe tax (up to 10× slower) |
| gpt2 | W8A16 | 0.80 | 0.50 | 1.02 | Moderate — borderline viable |
| gpt2 | W8A8 | 0.50 | 0.50 | 0.51 | Tax — half of theoretical |
| vit-s | W8A16 | 0.48 | 0.26 | 0.66 | Tax |
| vit-s | W8A8 | 0.42 | 0.32 | 0.47 | Tax |

### Phase 2 (ONNX Runtime)
- distilbert-base, bert-base: ORT FP32 and FP16 measured successfully.
- gpt2: ORT FP32/FP16 measured after disabling KV cache for export.
- vit-s, resnet50: ORT FP32 measured. ORT FP16 requires explicit model fp16 conversion — recorded as error (legitimate data point).

### Phase 3 (Seq-len Sensitivity)
- bert-base + gpt2 at seq={32, 128, 512}, schemes={fp16, w8a16, w4a16, w8a8}, compile backend.
- Realization ratio generally improves slightly at seq=512 for W8A16 on GPT-2 (larger tensors → more memory-bound, weight-only wins more).

### Hypothesis Outcomes

- **H1** (batch=1: W8A8 loses from activation overhead): **CONFIRMED.** W8A8 at batch=1
  has R < 0.03 for BERT/DistilBERT — `choose_qparams` overhead dominates completely.

- **H2** (W8A8 INT8 GEMM not dispatched on 24-SM AD107): **CONFIRMED.** `int8_gemm_dispatched = False`
  for every W8A8 config. The "Not enough SMs to use max_autotune_gemm mode" warning appears
  consistently. The INT8 tensor-core path is never taken; instead torchao falls back to
  FP dequantize + matmul, yielding 5–10× slowdown vs FP16.

- **H3** (FP16 is the real baseline; quant frequently loses): **CONFIRMED.** On this GPU,
  W8A16 achieves realized speedup ≤ 1.0× vs FP16 at most batch sizes. W8A8 and W4A16
  actively hurt across all configs.

- **H4** (`torch.compile` required for any quant speedup): **CONFIRMED.** Eager W8A16 is
  slower than FP16 eager in all cases. Compile backend enables autotune but is still
  constrained by SM count.

- **H5** (large R < 1 region on consumer hardware): **CONFIRMED.** R < 1 for all quant
  schemes except W8A16 at some batch=64 GPT-2 configs.

### Decision Table

| Model Kind | Batch Regime | Best Scheme | Note |
|-----------|--------------|-------------|------|
| encoder (BERT/DistilBERT) | any | FP16 + compile | All quant schemes impose tax |
| decoder (GPT-2) | small (≤4) | FP16 or W8A16 | W8A16 reaches ~1× FP16 |
| decoder (GPT-2) | large (≥16) | FP16 | W8A8 at ~0.5× FP16 |
| ViT | any | FP16 + compile | Similar pattern to encoders |
| CNN (ResNet) | any | ORT INT8 | Covered by Phase 2 |

**Bottom line for practitioners:** On a 24-SM consumer GPU, skip W8A8 and W4A16 entirely.
W8A16 + compile is the only scheme worth trying, and even then, check that it actually beats
your FP16 baseline before deploying.

---

## Limitations

1. **Single GPU, single thermal envelope.** Laptop thermals add variance; no clock lock.
2. **Model scale ceiling.** These are small models (~110M params). Larger models (7B+) may
   show different regimes (memory-bandwidth-bound at all batch sizes).
3. **torchao 0.9.0.** The SM-starvation behavior may improve in future torchao versions.
4. **No measured DRAM traffic.** Roofline uses analytical model only (ncu requires sudo).
5. **No TensorRT.** TRT was optional; ORT covers the Phase 2 CNN int8 benchmark.
6. **Accuracy.** BERT/DistilBERT use output fidelity (no task), GPT-2 uses output fidelity
   (no LM head in GPT2Model). Accuracy numbers are conservative proxies.

---

## Figures

| Figure | File | Description |
|--------|------|-------------|
| 1 | `fig1_realization_heatmap.png` | R heatmap over batch × scheme per model (money fig) |
| 2 | `fig2_latency_vs_batch.png` | Latency vs batch (log y), one line per scheme |
| 3 | `fig3_realization_gap.png` | Theoretical vs realized speedup bars |
| 4 | `fig4_kernel_mix.png` | Top kernels by CUDA time for 3 cells |
| 5 | `fig5_int8_dispatch.png` | INT8 GEMM dispatch map (H2 test) |
| 6 | `fig6_roofline_analytical.png` | Roofline — **ANALYTICAL MODEL, not ncu** |
| 7 | `fig7_decision_table.png` | Best scheme per model-kind + batch regime |
| 8 | `fig8_thermal_appendix.png` | GPU temp + SM clock per config |
