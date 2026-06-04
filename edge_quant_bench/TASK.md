# TASK: Edge-GPU Quantization Realization Benchmark (Idea A)

> **This is a self-contained task brief. You (the executing Claude session) have no prior
> context — everything you need is here. Read the whole document before writing code.**

---

## 0. Mission (one paragraph)

Build and run a rigorous inference-latency benchmark that answers: **on a consumer/edge GPU
(RTX 4060 Laptop, 8 GB, Ada sm_89, limited SMs), under what conditions does quantization
actually deliver its theoretical speedup — and where does it instead impose a "quantization
tax"?** The deliverable is a *map* of the realization ratio (realized speedup ÷ theoretical
speedup) across {model × quant scheme × backend × batch size}, plus kernel-level explanations
and a practitioner decision table. This is a **deployment-latency systems study**, NOT a
QAT-method paper and NOT a training study. Measure *deployed* models, with **FP16 as the
baseline to beat**.

**Two measurement layers, both first-class:** (1) **latency** = the realized "what" (the
headline metric a practitioner feels); (2) **deterministic metrics** (kernel count, kernel mix,
FLOPs, weight bytes, peak footprint, and — on a subset — DRAM traffic/occupancy via Nsight
Compute) = the "why" that *explains* the realization ratio. Deterministic metrics are collected
on **every** config, not as a fallback. They anchor the (thermally noisy) latency and provide
the mechanistic evidence for the hypotheses (esp. H2: is the INT8 GEMM actually dispatched?).

---

## 1. Environment (verify first, do not assume)

- Working dir for this task: `edge_quant_bench/` inside the repo at
  `/home/shreya/Coding/optimizer_V2`.
- Python venv (existing): `/home/shreya/venvs/fusion_qat/bin/python` — Python 3.12.
- Known-installed: `torch==2.6.0+cu124`, `torchao==0.9.0`, `transformers`, `datasets`.
- GPU: **NVIDIA GeForce RTX 4060 Laptop GPU**, 8 GB, Ada (sm_89). Single GPU. Laptop = WILL
  thermally throttle; you MUST control for this (Section 5).
- OS: Linux, bash. `nvidia-smi` available. `sudo` may or may not be available — detect it,
  degrade gracefully.

**Before anything else**, write and run `env_check.py` that prints and saves to
`results/env.json`: python exe, torch/torchao/transformers/datasets/onnxruntime versions,
`torch.version.cuda`, GPU name, total/free VRAM, SM count, current SM clock and temperature,
whether `sudo nvidia-smi` works. Stop and report if torch CUDA is unavailable.

Extra deps to install into the venv (pin if possible):
```bash
/home/shreya/venvs/fusion_qat/bin/python -m pip install onnxruntime-gpu timm
# onnx export: optimum may help for HF->onnx; install only if you choose that route
```

---

## 2. Background & framing (so you don't repeat known mistakes)

- A prior project in this repo (`experiments/`) **mis-measured** quantization by profiling the
  fake-quantize QAT *training* graph as if it were a deployment target, and by using a
  `StaticScaleLinear` that still ran FP32 matmuls. **Do not import from `experiments/` or
  reuse its results.** Treat it as historical only.
- Real, deployed quantization in `torchao` is applied via `quantize_(model, <config>())` and
  **only realizes speedups under `torch.compile`**. Eager quant is usually slower.
- On this exact GPU, a prior run showed `int8_weight_only + torch.compile` reaching ~7.8 ms
  vs FP32 ~11.8 ms (DistilBERT, batch 8, seq 64), while the QAT `.convert()` path was ~24 ms
  (slower than FP32), and the compiler warned `Not enough SMs to use max_autotune_gemm`. That
  SM-starvation effect is precisely the phenomenon this study characterizes.

---

## 3. Research question & preregistered hypotheses

**RQ:** Map the realization ratio `R = realized_speedup / theoretical_speedup`, where
`realized_speedup = latency(FP16) / latency(scheme)`, across the operating space. R ≥ 1 = quant
pays off; R < 1 = tax; `latency(scheme) > latency(FP16)` = quant actively hurts.

Record results for/against each hypothesis (do not tune to confirm them):
- **H1:** At batch=1 (memory/launch-bound), weight-only quant (W8A16, W4A16) gives real
  speedup, but dynamic-activation W8A8 loses (activation `choose_qparams` overhead unamortized).
- **H2:** At large batch (compute-bound), W8A8 INT8 tensor-core matmul wins only if the INT8
  GEMM kernel is actually dispatched; on AD107's limited SMs it often is not → realized ≪
  theoretical.
- **H3:** FP16 is the real baseline; quant frequently loses to it at this model scale.
- **H4:** `torch.compile` is required to realize any quant speedup vs eager FP16.
- **H5:** There is a large, characterizable region where R < 1 on consumer hardware.

---

## 4. Deliverable structure

```
edge_quant_bench/
  TASK.md                  # this file
  README.md                # written by you: how to reproduce + findings summary
  requirements.txt         # pinned deps you actually used
  env_check.py
  config.py                # the sweep matrix + constants (single source of truth)
  thermal.py               # clock lock (if sudo) + temp polling + cooldown
  models.py                # name -> dict(model, sample_input_fn(batch,seq), kind, eval)
  quantize.py              # scheme_name -> returns deployed model (applies quantize_, dtype)
  backends.py              # wrap model as callable: eager / compile / onnxruntime
  measure.py               # timing harness (latency stats, throughput, peak mem)
  deterministic.py         # cheap always-on metrics: kernel count+mix, FLOPs, weight bytes
  profile_ncu.py           # Nsight Compute subset: DRAM traffic, occupancy, roofline
  accuracy.py              # fidelity / accuracy guard rail
  run_sweep.py             # driver: iterate config matrix -> results/raw/<id>.json
  analyze.py               # aggregate raw -> results/results.parquet + realization ratio
  plots.py                 # heatmaps, latency-vs-batch, realization-gap, decision table
  results/
    env.json
    raw/                   # one json per (model,scheme,backend,batch,seq) config
    results.parquet
    figures/
  logs/
```

Commit to git at each phase milestone (branch off `main`, do not commit to `main` directly).

---

## 5. Measurement methodology (credibility lives here)

**Timing:** `torch.cuda.Event` with `synchronize()`. Always `model.eval()` +
`torch.inference_mode()`. Set `torch.set_float32_matmul_precision('high')`. Use fixed input
tensors per config (same data across schemes).

**Per-config protocol:**
1. Warmup 30 iters (triggers compile + cudnn autotune). Record compile/first-iter time
   **separately** from steady-state latency.
2. Measure 3 blocks × 100 iters. Record **mean, std, p50, p90, p99** (ms).
3. **Cooldown between every config:** via `thermal.py`, poll `nvidia-smi` and wait until GPU
   temp < 55 °C (cap the wait at, say, 60 s; if never reached, log it). Record temp_start,
   temp_end, mean SM clock, mean power for the config.

**Clock control (`thermal.py`):** if `sudo nvidia-smi` works, persistence mode on + lock SM
clock + cap power for stable numbers; record the locked values. If no sudo, **do not lock —
instead log SM clock every config and report it**, and rely on cooldown gating. The README must
state which mode was used.

**Reference timing core (adapt as needed):**
```python
import torch, statistics as st
def time_model(run_fn, warmup=30, blocks=3, iters=100, cooldown=lambda: None):
    with torch.inference_mode():
        for _ in range(warmup): run_fn()
        torch.cuda.synchronize()
        all_t = []
        for _ in range(blocks):
            for _ in range(iters):
                s, e = torch.cuda.Event(True), torch.cuda.Event(True)
                s.record(); run_fn(); e.record(); torch.cuda.synchronize()
                all_t.append(s.elapsed_time(e))
            cooldown()
    all_t.sort()
    pct = lambda p: all_t[min(len(all_t)-1, int(len(all_t)*p/100))]
    return dict(mean=st.mean(all_t), std=st.pstdev(all_t),
                p50=pct(50), p90=pct(90), p99=pct(99), n=len(all_t))
```

**Record per config (the row schema):** model, scheme, backend, batch, seq, dtype,
latency {mean,std,p50,p90,p99}, throughput_samples_per_s, peak_mem_mb,
**kernel_count, kernel_mix (list of {name,count,cuda_time_ms}), int8_gemm_dispatched (bool),
flops, weight_bytes, analytical_mem_bytes**, compile_time_s, temp_start, temp_end,
sm_clock_mhz, power_w, oom (bool), accuracy_metric, accuracy_value, timestamp, git_commit.
Save each row immediately to `results/raw/<id>.json` (crash-safe; resumable — skip configs
whose raw file already exists).

### 5a. Deterministic metrics (always-on, the "why" layer) — `deterministic.py`

Collected for **every** config alongside latency. These are reproducible and immune to
thermal throttling, so they anchor the latency numbers and directly test the hypotheses.

Be precise about what is actually deterministic and how to get it:

| Metric | Source | Deterministic? | Cost |
|---|---|---|---|
| FLOPs | analytical from model + input shapes (e.g. `fvcore`/`torch` or hand-derived) | fully | free |
| weight_bytes | param count × dtype bytes (post-quant effective size) | fully | free |
| analytical_mem_bytes | derived traffic model: weight bytes + activation bytes per layer | fully (approx) | free |
| peak_mem_mb | `torch.cuda.max_memory_allocated` (footprint, NOT traffic) | yes | free |
| kernel_count | torch profiler (CUDA events) | yes, given fixed compiled graph | cheap |
| kernel_mix | torch profiler op names + counts + cuda_time | yes | cheap |
| **int8_gemm_dispatched** | scan kernel names for INT8/cutlass GEMM vs FP fallback | yes | cheap |

`int8_gemm_dispatched` is the **direct test of H2** — for W8A8/W4A16 configs, did a real
low-precision tensor-core GEMM actually run, or did it fall back to FP + dequant? Flag it.

**Important caveat (do not misreport):** PyTorch's profiler does NOT give DRAM bytes moved.
`peak_mem_mb` is resident footprint, not traffic. True memory traffic comes from either the
analytical model (above, approximate) or Nsight Compute (Section 5b, exact). State this
distinction explicitly in the README.

### 5b. Nsight Compute roofline subset — `profile_ncu.py`

`ncu` is exact and deterministic but slow and may need elevated permissions, so run it on a
**representative subset (~6–10 cells)**, NOT the full sweep. Pick cells that span the regimes:
a clear quant win, a clear quant tax, batch=1 vs batch=64, W8A16 vs W8A8.

Detect availability first:
```bash
which ncu && ncu --version    # if missing or permission-denied, skip this layer + note in README
```
Per chosen cell capture: **dram__bytes (read+write), achieved occupancy, sm__throughput,
arithmetic intensity** → enough for a roofline plot. Save to `results/ncu/<id>.json`. If `ncu`
is unavailable, fall back to the analytical traffic model and clearly label figures as
analytical, not measured.

---

## 6. Quantization schemes (exact torchao 0.9.0 APIs)

`quantize_` mutates `nn.Linear` modules in place; apply, then `torch.compile`.
```python
from torchao.quantization import (quantize_, int8_weight_only, int4_weight_only,
                                   int8_dynamic_activation_int8_weight)
# FP32: as-is. FP16: model.half(). BF16: model.bfloat16().
# W8A16: quantize_(model, int8_weight_only())
# W4A16: model.bfloat16(); quantize_(model, int4_weight_only(group_size=128))  # tinygemm needs bf16
# W8A8 : quantize_(model, int8_dynamic_activation_int8_weight())
```
**Gotchas to handle (and record, don't hide):**
- `int4_weight_only` requires bf16 activations (tinygemm). If a model/shape errors, log it as a
  failed cell — failures are data.
- torchao weight-only/dyn-act target **`nn.Linear` only**. **ResNet-50 is conv-heavy** →
  torchao will quantize almost nothing. Therefore: **Phase 1 (torchao) uses the 4 linear-heavy
  models only** (DistilBERT, BERT-base, GPT-2, ViT-S). Put ResNet-50 and CNN int8 in **Phase 2
  via ONNX Runtime / TensorRT** (which support conv int8). State this clearly in README.
- 8 GB VRAM: large batch×seq may OOM. Catch OOM, record `oom=true`, `torch.cuda.empty_cache()`,
  continue. Do not crash the sweep.

---

## 7. Models (`models.py`)

| name | source | kind | notes |
|---|---|---|---|
| distilbert-base | `transformers` distilbert-base-uncased | encoder | seq sweep applies |
| bert-base | `transformers` bert-base-uncased | encoder | seq sweep applies |
| gpt2 | `transformers` gpt2 | decoder | seq sweep applies; PPL eval |
| vit-s | `timm` vit_small_patch16_224 | vit | fixed 224²; batch sweep |
| resnet50 | `timm` resnet50 | cnn | Phase 2 (ONNX/TRT) only |

Each entry exposes: the model (pretrained, eval), a `sample_input(batch, seq)` builder on CUDA,
its `kind`, and an `eval` hook (Section 8). Use realistic dummy inputs (right dtype/range);
real task data only where accuracy is measured.

---

## 8. Accuracy / fidelity guard rail (`accuracy.py`)

Purpose: disqualify "fast but broken" quant; report speedup only for valid models.
- **GPT-2:** WikiText-2 perplexity (real, meaningful). Flag if PPL rises > 10% vs FP16.
- **ViT-S, ResNet-50:** ImageNet-1k val **top-1 on a ~1–2k subset** if available; if ImageNet
  is not present, fall back to output-fidelity-vs-FP16 and note it.
- **BERT/DistilBERT (base, not fine-tuned):** absolute task accuracy is meaningless, so use
  **output fidelity vs the FP16 model**: cosine similarity + max-abs-diff of final logits over
  a fixed 256-sample input set. Flag if cosine < 0.99.
Record the metric name + value in every row.

---

## 9. Sweep matrix & phasing (`config.py`)

**Phase 1 — core map (torchao, the paper's backbone):**
- models: distilbert-base, bert-base, gpt2, vit-s
- schemes: FP32, FP16, BF16, W8A16, W4A16, W8A8
- backends: eager, compile
- batch: 1, 4, 16, 64   (NLP seq fixed = 128; vit fixed 224²)
- ≈ 4×6×2×4 = 192 configs. This alone is publishable.

**Phase 2 — backends & CNN:** add ONNX Runtime (CUDA EP) for all models incl. resnet50; add
INT8 via ORT for conv. (TensorRT EP optional if it installs cleanly.)

**Phase 3 — sensitivity:** seq-len sweep {32,128,512} on bert-base + gpt2; kernel-mix deep
dive on 3–4 representative win/loss cells.

---

## 10. Execution order with GO/NO-GO gates

1. **Phase 0 — harness validation (do this before the full sweep).**
   - Build env_check, thermal, measure, models, quantize, backends.
   - Sanity run a *tiny* subset: bert-base @ {FP32, FP16} × {eager, compile} × batch{1,16}.
   - Verify `deterministic.py` produces stable kernel_count / kernel_mix / int8_gemm_dispatched
     and that `flops`/`weight_bytes` are sane. Confirm `ncu` availability (or note absence).
   - **GATE (latency quality only):** FP16 must be reproducibly faster than FP32; per-config
     std must be a small fraction of mean (target std/mean < ~5% after cooldown). If variance is
     too high even with cooldown + (optional) clock lock → **report to the user**, but **do NOT
     abandon the project**: the deterministic + ncu layers (kernel mix, dispatch, DRAM traffic,
     roofline) are throttling-immune and still carry the paper. In that case, demote latency to
     a noisy-but-reported secondary metric and lead with the deterministic mechanism. The
     deterministic metrics are collected regardless of how this gate resolves.
2. **Phase 1 sweep** → `results/raw/`. Then `analyze.py` + first heatmaps. Report findings.
3. **Phase 2** (ONNX/CNN). 4. **Phase 3** (seq-len + kernel deep dive).
5. Write `README.md`: hardware, method, the realization-ratio map, the decision table, and
   which hypotheses held.

After each phase: commit, then give the user a short findings summary and the key figure paths.

---

## 11. Figures/tables to produce (`plots.py`)

1. Heatmap grid: R over (batch × scheme), one panel per model (compile backend). **Money fig.**
2. Latency-vs-batch line plots per model (log y), one line per scheme.
3. Realization-gap bars: theoretical vs realized speedup, shortfall highlighted.
4. Kernel-mix breakdown for 2–3 representative cells (explains the "why").
5. **INT8-GEMM-dispatch map:** per (model × scheme × batch), did a real low-precision tensor-core
   GEMM run? (binary heatmap) — the direct visual test of H2.
6. **Roofline plot** from the `ncu` subset: representative cells on the AD107 roofline, showing
   memory-bound vs compute-bound and where quant moves them.
7. Decision table: best scheme per (model-kind, batch regime).
8. Appendix: temp/clock per config (proves results aren't throttling artifacts).

For theoretical speedup use a documented, simple model (e.g., weight-bytes ratio for
memory-bound small-batch; bit-width/throughput ratio for compute-bound large-batch) and state
the assumption explicitly in README.

---

## 12. Definition of done

- Resumable sweep writes one JSON per config; rerun skips completed cells.
- Every row carries the deterministic metrics (kernel count/mix, int8_gemm_dispatched, FLOPs,
  weight bytes, analytical traffic) alongside latency; `ncu` subset captured (or absence noted).
- `results/results.parquet` aggregates all rows; `analyze.py` computes R with FP16 baseline.
- All 8 figures render from saved results with one `python plots.py`.
- README documents environment, clock-control mode, methodology, the map, the decision table,
  hypothesis outcomes, and limitations (single GPU, laptop thermals, model-scale ceiling).
- Everything reproducible from the venv with pinned `requirements.txt`.

---

## 13. How to work / interaction rules

- Work incrementally; validate Phase 0 gate before the big sweep.
- Decide routine engineering autonomously. **Ask the user only at:** the Phase 0 variance gate
  (if it fails), missing data dependencies (e.g., ImageNet), or if a core dependency
  (onnxruntime-gpu/TensorRT) won't install.
- Never fabricate numbers. If a cell fails/OOMs/kernel-missing, record it as a result.
- Keep the dead `experiments/` project out of scope.
- Branch off `main`; commit per phase with clear messages.
```
Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>
```
```
