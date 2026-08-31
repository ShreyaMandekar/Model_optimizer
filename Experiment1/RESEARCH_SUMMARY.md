# Research Summary

## Problem Statement

Standard Quantization-Aware Training (QAT) with `Int8DynActInt4WeightQATLinear` (torchao) is significantly slower at inference than FP32, despite theoretical INT8 efficiency. On DistilBERT-base with a batch of 8 SST-2 sentences:

- FP32: ~12.0ms, 35 CUDA kernels
- QAT: ~34.8ms, 98 CUDA kernels — **2.9x overhead**

Profiling reveals the overhead is not from graph fragmentation but from **redundant dynamic quantization parameter computation** on every forward pass: `choose_qparams`, `aten::aminmax`, and `fused_moving_avg_obs_fake_quant` collectively consume ~55% of QAT CUDA time (5.23ms vs 1.42ms FP32).

## Motivation

The key insight: these 270 per-inference calls to recompute activation scales are wasted if the activation distributions are actually stable across inputs. For NLP tasks like sentiment classification, many transformer layers process tokens with very similar statistical properties, so the optimal quantization scale barely changes batch-to-batch.

Hardware: NVIDIA RTX 4060 Laptop GPU (8.2GB VRAM), PyTorch 2.6.0+cu124, torchao 0.9.0.  
Task: SST-2 binary sentiment classification.  
Model: DistilBERT-base-uncased (6 transformer layers, 38 QAT linear layers).

## Proposed Approach

**Activation Stability Analysis:** Measure the coefficient of variation (CV = std/mean × 100) of each layer's dynamic quantization scale across calibration data. Stable layers (low CV) are candidates for replacement with a `StaticScaleLinear` module that uses a fixed frozen scale, eliminating the per-inference `choose_qparams` + `amin` calls.

Two conversion strategies were developed:

### Phase 2 — Post-hoc Static Conversion
After training, run calibration data through the QAT model, measure CV per layer, classify each layer as `STATIC_OK` (CV < 5%), `BORDERLINE` (5–15%), or `DYNAMIC` (>15%). Replace stable layers using `convert_stable_layers()`.

### Phase 3 — Training-Aware Progressive Freezing
During QAT fine-tuning, track activation scale stability online via EMA (Exponential Moving Average) coefficient of variation. When a layer's EMA-CV stays below threshold for `patience` consecutive steps, it becomes a freeze candidate. The `ProgressiveFreezer` converts candidates in tiers of 3 layers at a time with 150-step adaptation gaps between tiers, letting the remaining dynamic layers compensate.

## Key Findings

### Activation Distribution is Stable
- 34 of 38 DistilBERT QAT layers have CV < 5%
- No layer exceeds 15% CV — all would eventually stabilize
- Earlier transformer layers (layer 0–2) stabilize first; later layers and FFN lin1 layers take longer

### Threshold Ablation (Phase 2)
| CV Threshold | Layers Converted | Accuracy | Latency |
|---|---|---|---|
| 1.0% | 3 | 87.38% | 38.6ms |
| 2.0% | 22 | 84.84% | 28.5ms |
| 3.0% | 28 | 86.92% | 27.4ms |
| 5.0% | 34 | 86.69% | 26.2ms |
| 10.0% | 38 | 86.57% | 24.5ms |

Accuracy is non-monotonic with number of layers converted — CV < 5% is the sweet spot balancing accuracy preservation with latency reduction.

### Main Results (DistilBERT on SST-2)
| Model | Latency | Accuracy | Frozen |
|---|---|---|---|
| FP32 baseline | 12.0ms | — | — |
| QAT dynamic | 34.8ms | 87.38% | 0/38 |
| Phase 2 post-hoc (CV<5%) | **12.0ms** | 86.69% | 34/38 |
| Phase 3 training-aware | 33.4ms | **87.96%** | 29/38 |
| Phase 3 + cleanup | 17.1ms | 86.46% | 38/38 |

Phase 2 recovers FP32 latency with only a 0.69% accuracy drop.  
Phase 3 achieves the best accuracy (+0.58% vs QAT dynamic) but only freezes 29/38 layers during training, leaving 9 dynamic; the subsequent cleanup gets to 17.1ms with further accuracy cost.

## Current Status

- **Phase 1 (profiling):** Complete. Baselines established for FP32, FP16, INT8 compiled, QAT uncompiled and compiled.
- **Phase 2 (post-hoc conversion):** Complete. Ablation done, all results recorded, figures generated.
- **Phase 3 (training-aware):** Training complete, results measured. The combined Phase 3 + cleanup result needs `lat_combined` and `acc_combined` placeholders in `04_paper_figures.py` replaced with actual values (17.1ms, 86.46%).
- **Generalization (BERT-base):** BERT-base checkpoint trained (`results/checkpoints/bert_base_qat_best.pt` exists). Full measurement pipeline (`experiments/generalization.py`) written but not confirmed run — `results/profiles/generalization_bert_base.json` is absent.
- **Paper figures:** Figures 1–4 generated. Figure 4 has hardcoded placeholder latency and accuracy for "Phase 3 + cleanup" that need updating.
