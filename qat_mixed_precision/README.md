# Online KL-Guided Mixed-Precision QAT

Turns the KL-Lens sensitivity signal from a **post-hoc, static** mixed-precision
tool into a **closed-loop, training-time** controller, and tests whether the
resulting **co-adaptation** beats post-hoc KL-Lens allocation at matched average
bit-width. The paper this code supports lives in `../paper_mixed_precision/`;
`../PAPER_VS_CODE_AUDIT.md` documents how the two were reconciled and is the
best "how does this all fit together" read if you're new to the repo.

## What's here

| file | role |
|---|---|
| `config.py` | single source of truth: hyperparameters, budget schedule, controller thresholds, the run list |
| `models.py` | loads a HF checkpoint (mamba-130m / mamba2-130m / gpt2) and wraps its projection layers |
| `data.py` | WikiText loaders: block-tokenized LM data, calib batch, perplexity eval |
| `dyn_precision_linear.py` | STE fake-quant linear, settable weight bits {2,4,8}; handles `nn.Linear` (Mamba) and `Conv1D` (GPT-2) |
| `marginal_kl.py` | forward-only marginal-KL audit (student→teacher), referenced to FP teacher |
| `precision_controller.py` | budgeted allocator: anneal 8→B\*, greedy drop, KL-triggered recovery |
| `train_qat.py` | one run; methods: `uniform` / `posthoc_base` / `posthoc` / `online` |
| `env_check.py` | verifies CUDA/package versions, saves `results/env.json` |
| `run_all.py` | crash-safe sweep, multi-GPU dispatch, skip-if-done |
| `analyze.py` | aggregates runs -> `results/analysis.parquet`; RQ1 paired test + the paper's sub-4-bit bootstrap-CI stat |
| `plots.py` | the paper's three figures: Pareto (`fig_pareto`), co-adaptation trajectory (`fig_traj`), per-projection allocation (`fig_allocation`) |
| `submit.sh` | PARAM Shavak entrypoint (env → unit tests → wrap-count sanity → Phase-0 gate → sweep → analysis/plots) |
| `tests/` | `pytest` unit tests (currently: `DynPrecisionLinear` quant/dequant correctness) |

## Run on PARAM Shavak

```bash
cd qat_mixed_precision
# (recommended, for Mamba speed; needs CUDA toolkit/nvcc)
pip install mamba-ssm causal-conv1d
bash submit.sh                # full pipeline on all visible GPUs
# or: python run_all.py --gpus 0,1
python run_all.py --dry       # show the plan without running
```

Tunables via env (defaults match the paper's Table `tab:hparams` — see
`../PAPER_VS_CODE_AUDIT.md` Part 1): `QMP_TOTAL_STEPS` (1500), `QMP_SEQ_LEN`
(512), `QMP_TRAIN_BATCH` (8), `QMP_CALIB_SAMPLES` (8), `QMP_CALIB_SEQ_LEN`
(256), `QMP_CONSOLE_DIR`, `QMP_PYTHON`. (Console output goes to `console/`, not
`logs/` — PARAM Shavak forbids writing to any path containing "log".)

## The sweep (≈50 runs)

- **Primary** mamba-130m × 3 seeds: `uniform{2,4,8}`, `posthoc{B3,4,5,6}`,
  `online{B3,4,5,6}` (+ shared 8-bit base) = 36 runs.
- **Second model** mamba2-130m + **transfer** gpt2, 1 seed each ≈ 14 runs.
  As of the data in `results/`, only `gpt2` (B4/B6, seed0) has actually been
  run — see `../PAPER_VS_CODE_AUDIT.md` Part 4 before citing any B3 or
  multi-seed GPT-2 number.

**Est. time** (mamba-130m, batch 8, seq 512, 1500 steps): with `mamba_ssm`
proportionally less than the original seq-1024/4000-step estimate; without it,
budget ~3-5x longer per run than with it.

## Phase-0 gate

`submit.sh` prints the FP WikiText-2 perplexity for mamba-130m before the sweep
(KL-Lens FP16 reference ≈ 21.6). If it's much higher than ~30, stop and reconsider
the testbed. Report this number before launching the full sweep. Note this is a
*zero-shot* pretrained-checkpoint number, not comparable to the post-QAT-training
`uniform-8`/`posthoc_base` numbers in `results/` (~17.8 PPL after 1500 steps of
fine-tuning on WikiText-103).

## Outputs

`results/runs/<run_id>.parquet` (one per run, trajectory + final PPL/avg_bits),
`results/audits/<run_id>.jsonl` (controller/audit events),
`results/ckpts/<model>__seed<N>__base8.pt` (shared 8-bit base checkpoints, reused
by `posthoc`), `results/analysis.parquet`, `results/figures/{fig_pareto,fig_traj,
fig_allocation}.pdf` (+ PNG previews).

## Tests

```bash
pytest tests/ -v
```
