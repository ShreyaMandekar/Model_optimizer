# Project State

## What is Implemented

### Core Library (`src/`)

**`src/graph/boundary_detector.py` — `ActivationStabilityDetector`**
- Attaches forward hooks to all `Int8DynActInt4WeightQATLinear` modules
- Records activation scale (`x.abs().amax(dim=-1) / 127.0`) across calibration batches
- Computes per-layer CV (coefficient of variation)
- Classifies layers: STATIC_OK (CV < 5%), BORDERLINE (5–15%), DYNAMIC (>15%)
- Estimates latency savings based on hardcoded profiler numbers
- Status: **Working**

**`src/graph/static_converter.py` — `StaticScaleLinear`, `convert_stable_layers()`**
- `StaticScaleLinear`: drop-in replacement for `Int8DynActInt4WeightQATLinear` with a frozen scale buffer; uses fake-quantize forward (clamp + round + scale)
- `convert_stable_layers()`: navigates model tree by name, replaces stable QAT layers in-place
- Status: **Working**

**`src/training/online_stability_tracker.py` — `OnlineStabilityTracker`**
- EMA-based per-layer activation scale variance tracking during training
- Parameters: `cv_threshold=5.0`, `patience=20`, `ema_alpha=0.15`
- Declares a layer freeze-eligible after `patience` consecutive steps with EMA-CV below threshold
- Hooks skip already-frozen layers
- Status: **Working**

**`src/training/progressive_freezer.py` — `ProgressiveFreezer`**
- Converts the top `freeze_tier_size` stable candidates to `StaticScaleLinear` in-place
- Enforces `adaptation_steps` gap between freeze tiers
- Logs step, layer name, and frozen scale for each conversion
- Status: **Working**

**`src/training/trainer.py` — `FusionAwareTrainer`**
- Full training loop with stability tracking and progressive freezing
- AdamW optimizer, CrossEntropyLoss
- Validates at end of each epoch, saves best checkpoint
- Writes training log and freeze schedule to `results/profiles/`
- Status: **Working**

**`src/evaluation/profiler.py`**
- `profile_model()`: CUDA+CPU profiler with top-ops extraction
- `measure_latency()`: CUDA event timing over 100 runs
- `save_results()`, `print_comparison()`: reporting utilities
- Status: **Working**

---

## What is Working

### Phase 1 — Baseline Profiling
- FP32 / FP16 / INT8 / QAT profiling complete
- Results: FP32=12.0ms, FP16=4.41ms, INT8-compiled=7.82ms, QAT=34.8ms (compiled)
- Profiles saved: `results/profiles/fp32_profile.json`, `qat_profile.json`, `fp32_latency.json`, `qat_latency.json`
- Figures: `fig1_bert_comparison.png`, `fig2_bert_qdq_breakdown.png`, `fig3_bert_key_insight.png`

### Phase 2 — Post-hoc Static Conversion
- Calibration, CV measurement, and static conversion pipeline fully working
- Threshold ablation: 6 thresholds tested (1.0–15.0%), results saved
- Best result: CV<5%, 34/38 layers converted, 86.69% accuracy, **12.0ms latency**
- Ablation data: `results/profiles/threshold_ablation.json`
- Checkpoint: `results/checkpoints/qat_finetuned.pt`
- Figure 2 (CV distribution) and Figure 3 (ablation) generated and saved

### Phase 3 — Training-Aware Progressive Freezing
- Full training run completed (5 epochs, DistilBERT + SST-2)
- 29/38 layers frozen during training; best val acc = **87.96%**
- Full model saved: `results/checkpoints/phase3_full_model.pt`
- Best checkpoint: `results/checkpoints/phase3_best.pt`
- Logs: `results/profiles/phase3_training_log.json`, `phase3_freeze_schedule.json`

### Phase 3 + Cleanup
- Post-hoc calibration applied to 9 remaining dynamic layers
- Combined model: all 38 frozen, **17.1ms ± 0.2ms, 86.46%**
- Combined checkpoint: `results/checkpoints/phase3_combined_full.pt`

### Paper Figures
- Figures 1–4 generated as PDF + PNG in `results/figures/`

---

## What is Partially Implemented

### Generalization to BERT-base
- BERT-base QAT training: **done** (checkpoint `bert_base_qat_best.pt` exists)
- Training notebook (`05_generalization.ipynb`) ran 3 epochs, reached 86.69% val accuracy
- Stability analysis, static conversion, and latency measurement: **written** in `experiments/generalization.py` (full pipeline) and notebook
- **Not confirmed executed:** `results/profiles/generalization_bert_base.json` is absent — the measurement pipeline was written but the final run was not completed

---

## Known Issues

### Figure 4 Placeholder Values
`experiments/04_paper_figures.py` lines 161–162:
```python
lat_combined = 12.5   # REPLACE with actual
acc_combined = 87.96  # REPLACE with actual
```
Correct values (from `03_phase3_part3.ipynb` output): `lat_combined = 17.1`, `acc_combined = 86.46`.

### Phase 3 Incomplete Freezing
Only 29/38 layers freeze during the 5-epoch training run. The 9 unfrozen layers (`layer.5` attention out_lin, ffn layers, and `pre_classifier`) never accumulate `patience=50` consecutive stable EMA-CV steps. Possible causes:
- 5 epochs may be insufficient for these layers to stabilize with `patience=50`
- FFN `lin1` layers (large weight matrices) may be inherently less stable
- The adaptation step constraint (150 steps) could be limiting; at 312 batches/epoch, roughly 10 tiers are achievable in 5 epochs, which matches the 10 freeze events

### Generalization Results Missing
`results/profiles/generalization_bert_base.json` does not exist. The BERT-base generalization experiment training is done but the full measurement/comparison has not been run.

### `StaticScaleLinear` Fake-Quantize Only
The `StaticScaleLinear` forward still applies fake-quantization (quantize + dequantize) rather than true INT8 kernels. This means the inference speedup comes only from eliminating the `choose_qparams` call, not from INT8 arithmetic acceleration. Actual INT8 execution would require exporting to TorchScript or ONNX with proper INT8 kernel dispatch.

### Hardcoded Profiler Numbers in `boundary_detector.py`
Lines 85–86 of `src/graph/boundary_detector.py` contain hardcoded cost estimates (`ms_choose = 27.0 / 120`, `ms_amin = 14.0 / 120`) derived from specific profiler runs. These would need recalibration for different hardware or batch sizes.

---

## Environment and Dependencies

- Python 3.12.3
- PyTorch 2.6.0+cu124
- torchao 0.9.0
- transformers (HuggingFace)
- datasets (GLUE SST-2)
- CUDA 12.4, RTX 4060 Laptop GPU
- Virtual env: `/home/shreya/venvs/fusion_qat/`
