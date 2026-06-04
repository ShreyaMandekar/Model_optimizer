# Experiment Log

---

## Experiment 1 — Phase 1: Baseline Profiling
**File:** `experiments/01_baseline_distilbert.ipynb`

### Configuration
- Model: `distilbert-base-uncased` (DistilBertForSequenceClassification)
- Task: SST-2 binary sentiment classification (GLUE benchmark)
- Batch size: 8, sequence length: 64
- Hardware: RTX 4060 Laptop GPU, 8.2GB VRAM
- Framework: PyTorch 2.6.0+cu124, torchao 0.9.0
- Profiler: `torch.profiler` (CUDA + CPU activities), 5 warmup + 10 profile runs
- Latency: 20 warmup + 100 timed runs with CUDA events

### Results
| Model | CUDA Kernels | CUDA Time (ms) | Latency (mean ms) |
|---|---|---|---|
| FP32 | 35 | 1.42 | 11.82 |
| FP16 | 58 | — | 4.41 |
| INT8 (torchao weight-only, compiled) | 18 | — | 7.82 |
| QAT uncompiled | 98 | — | 45.92 |
| QAT compiled | 54 | 5.23 | ~34.8 |

### QAT Op Breakdown (CUDA time, compiled)
| Operation | CUDA Time | Count | Notes |
|---|---|---|---|
| fused_moving_avg_obs_fake_quant | 0.413ms | 270 | QDQ overhead |
| aten::aminmax | 0.148ms | 270 | QDQ overhead |
| choose_qparams | 0.076ms | 270 | QDQ overhead |
| fake_quantize_per_tensor | 0.063ms | 220 | QDQ overhead |
| aten::addmm | 0.052ms | 70 | Useful compute |
| aten::bmm | 0.044ms | 40 | Useful compute |

**Total QDQ overhead: ~55% of QAT CUDA time**

### Conclusions
- QAT is 2.9x slower than FP32 despite INT8 arithmetic, due to 270 per-inference dynamic scale computations
- FP16 fuses natively and achieves 2.68x speedup; QAT breaks compiler fusion
- INT8 weight-only (compiled) reaches 7.82ms — shows INT8 can be fast when done right
- The overhead target: eliminate `choose_qparams` + `amin` calls for stable layers

---

## Experiment 2 — Phase 2: Activation Stability Analysis and Post-hoc Conversion
**File:** `experiments/02_stability_analysis.ipynb`

### Configuration
- Model: DistilBERT-base QAT fine-tuned on SST-2 (checkpoint `results/checkpoints/qat_finetuned.pt`)
- Calibration data: 200 SST-2 validation sentences (25 batches of 8)
- Stability metric: Coefficient of Variation (CV = std/mean × 100) of activation scale per layer
- Activation scale computed as: `x.abs().amax(dim=-1) / 127.0`
- Thresholds: CV < 5% = STATIC_OK, 5–15% = BORDERLINE, >15% = DYNAMIC
- Fine-tuning: 3 epochs on SST-2 train (first 5000 samples), AdamW lr=2e-5

### CV Distribution (38 QAT layers)
Measured CV values across all 38 layers (data in `experiments/04_paper_figures.py` `cv_values` list):
- Range: 1.53% – 6.23%
- STATIC_OK (CV < 5%): 34/38 layers
- BORDERLINE (5–15%): 4/38 layers (layer 0 q/k/v attention, pre_classifier, classifier)
- DYNAMIC (>15%): 0/38 layers

**Key observation:** No layer exceeds 15% CV. The activation distribution is remarkably stable across inputs.

### Threshold Ablation (saved to `results/profiles/threshold_ablation.json`)
| CV Threshold | Layers Converted | Accuracy (%) | Latency (ms) |
|---|---|---|---|
| 1.0% | 3 | 87.38 | 38.63 |
| 2.0% | 22 | 84.84 | 28.54 |
| 3.0% | 28 | 86.92 | 27.43 |
| **5.0%** | **34** | **86.69** | **26.20** |
| 10.0% | 38 | 86.57 | 24.51 |
| 15.0% | 38 | 86.57 | 24.73 |

- Optimal threshold: **CV = 5%** — converts 34/38 layers, accuracy drop only 0.69%, 1.33x speedup over dynamic
- Accuracy is non-monotonic: converting 22 layers (threshold 2%) causes more accuracy loss than converting 34 (threshold 5%), suggesting scale sensitivity varies by layer
- 34/38 converted at 5% threshold: latency matches FP32 baseline (12.0ms in compiled/full measurements)

### Output Verification
- Max logit deviation after static conversion: checked to be small (output consistency test in notebook cell 14)
- Accuracy on full SST-2 validation: 86.69% (baseline QAT dynamic: 87.38%)

### Conclusions
- Most DistilBERT layers are activation-stable — post-hoc static conversion is safe
- CV < 5% threshold recovers FP32 latency with minimal accuracy loss
- The 4 borderline layers include early attention projections and the final classifier

---

## Experiment 3 — Phase 3: Training-Aware Progressive Freezing
**Files:** `experiments/03_phase3_training.py`, `experiments/03_phase3_part2.py`, `experiments/03_phase3_part3.ipynb`

### Configuration
- Base: DistilBERT with QAT layers applied, initialized from `qat_finetuned.pt` checkpoint
- Training data: SST-2 train (first 5000 samples), batch size 16
- Validation: full SST-2 validation set
- Optimizer: AdamW, lr=1e-5 (lower than Phase 2 fine-tuning)
- Epochs: 5
- Stability tracking: EMA-CV with alpha=0.15 (faster response than original 0.05)
- Freeze hyperparameters:
  - `cv_threshold = 5.0%`
  - `patience = 50` consecutive stable steps
  - `freeze_tier_size = 3` layers per tier
  - `adaptation_steps = 150` steps between tier freezes

### Freeze Schedule (`results/profiles/phase3_freeze_schedule.json`)
| Step | Layers Frozen | Layer Names |
|---|---|---|
| 149 | 3 | layer.0.attention.out_lin, layer.0.ffn.lin1, layer.0.ffn.lin2 |
| 299 | 3 | layer.1.attention.q/k/v_lin |
| 449 | 3 | layer.2.attention.q/k/v_lin |
| 599 | 3 | layer.3.attention.q/k/v_lin |
| 749 | 3 | layer.4.attention.out_lin, layer.4.ffn.lin1, layer.1.ffn.lin2 |
| 899 | 3 | layer.1.attention.out_lin, layer.2.ffn.lin1, layer.1.ffn.lin1 |
| 1049 | 3 | layer.4.attention.q/k/v_lin |
| 1199 | 3 | layer.3.ffn.lin1, layer.4.ffn.lin2, layer.2.ffn.lin2 |
| 1349 | 3 | layer.5.attention.q/k/v_lin |
| 1499 | 2 | layer.3.ffn.lin2, classifier |

**Total frozen after training: 29/38 layers**

### Training Results (`results/profiles/phase3_training_log.json`)
- Val accuracy progression: 86.34% → 87.96% → 86.81% → 86.11% → 87.85%
- Best val accuracy: **87.96%** (epoch 2), saved to `results/checkpoints/phase3_best.pt`
- Training loss: low (0.02–0.16 range), model converges well
- Frozen count at end: 29/38 — 9 dynamic layers remaining

### Phase 3 + Cleanup (Part 3)
After training, Phase 2 calibration applied to the 9 remaining dynamic layers:
- All 9 remaining layers converted to static via post-hoc calibration
- **Combined result: 17.1ms ± 0.2ms, 86.46% accuracy, 38/38 frozen**
- Model saved to `results/checkpoints/phase3_combined_full.pt`

### Complete Results Table (from `03_phase3_part3.ipynb`)
| Model | Latency | Accuracy | Frozen |
|---|---|---|---|
| FP32 baseline | 12.0ms | — | — |
| QAT dynamic | 34.8ms | 87.38% | 0/38 |
| Phase 2 post-hoc | 12.0ms | 86.69% | 34/38 |
| Phase 3 training-aware | 33.4ms | 87.96% | 29/38 |
| Phase 3 + cleanup | 17.1ms | 86.46% | 38/38 |

### Conclusions
- Training-aware freezing achieves better accuracy (87.96%) than both dynamic QAT and post-hoc conversion, validating the hypothesis that remaining layers adapt to compensate for frozen ones
- Phase 3 alone only freezes 29/38 during training; 9 layers never meet the patience criterion
- Full freezing via Phase 3 + cleanup achieves 17.1ms (2.04x speedup over dynamic QAT) but loses 1.5% accuracy vs best Phase 3 model
- Phase 2 post-hoc remains the best pure latency result (12.0ms = FP32) at only 0.69% accuracy cost

---

## Experiment 4 — Generalization: BERT-base-uncased
**Files:** `experiments/05_generalization.ipynb`, `experiments/generalization.py`

### Configuration
- Model: `bert-base-uncased` (BertForSequenceClassification), 12 transformer layers
- Task: SST-2 (same as above)
- Training data: 5000 SST-2 train samples (notebook), 10000 in `generalization.py`
- Epochs: 3 (notebook), 5 (script)
- Optimizer: AdamW lr=2e-5

### Training Results (from notebook cell outputs)
| Epoch | Val Accuracy |
|---|---|
| 1 | 81.83% |
| 2 | 86.46% |
| 3 | 86.69% |

Checkpoint saved: `results/checkpoints/bert_base_qat_best.pt`

### Status: Incomplete
- BERT-base training completed (checkpoint exists)
- Stability analysis, static conversion, and latency measurement pipeline written in `generalization.py`
- `results/profiles/generalization_bert_base.json` not yet generated — pipeline not run to completion
- Expected: BERT-base has ~144 QAT linear layers; if similar CV distribution holds, substantial latency reduction is achievable

---

## Experiment 5 — Paper Figure Generation
**File:** `experiments/04_paper_figures.py`

### Figures Generated
- **Fig 1** (`fig1_profiler_breakdown.pdf/png`): QAT vs FP32 CUDA op breakdown — shows 55% quantization overhead
- **Fig 2** (`fig2_cv_distribution.pdf/png`): Histogram of CV values for all 38 layers — 34 below 5% threshold
- **Fig 3** (`fig3_threshold_ablation.pdf/png`): Pareto curve (latency vs accuracy) for threshold sweep + layers-converted vs accuracy
- **Fig 4** (`fig4_main_results.pdf/png`): Bar charts comparing latency and accuracy for all 5 model variants

### Known Issue in Fig 4
`lat_combined` and `acc_combined` are hardcoded as placeholder values (12.5ms, 87.96%) in `04_paper_figures.py`. These should be updated to the measured values: **17.1ms, 86.46%**.
