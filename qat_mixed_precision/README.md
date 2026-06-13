# Online KL-Guided Mixed-Precision QAT

Turns the KL-Lens sensitivity signal from a **post-hoc, static** mixed-precision
tool into a **closed-loop, training-time** controller, and tests whether the
resulting **co-adaptation** beats post-hoc KL-Lens allocation at matched average
bit-width. Full brief: **`TASK.md`** (read it first).

## What's here

| file | role |
|---|---|
| `dyn_precision_linear.py` | STE fake-quant linear, settable weight bits {2,4,8}; handles `nn.Linear` (Mamba) and `Conv1D` (GPT-2) |
| `marginal_kl.py` | forward-only marginal-KL audit (student→teacher), referenced to FP teacher |
| `precision_controller.py` | budgeted allocator: anneal 8→B\*, greedy drop, KL-triggered recovery |
| `train_qat.py` | one run; methods: `uniform` / `posthoc_base` / `posthoc` / `online` |
| `run_all.py` | crash-safe sweep, 2-GPU dispatch, skip-if-done |
| `analyze.py` / `plots.py` | Pareto (PPL vs bits), RQ1 paired test |
| `submit.sh` | PARAM Shavak entrypoint (env → unit test → Phase-0 gate → sweep) |

## Run on PARAM Shavak

```bash
cd qat_mixed_precision
# (recommended, for Mamba speed; needs CUDA toolkit/nvcc)
pip install mamba-ssm causal-conv1d
bash submit.sh                # full pipeline on all visible GPUs
# or: python run_all.py --gpus 0,1
python run_all.py --dry       # show the plan without running
```

Tunables via env: `QMP_TOTAL_STEPS`, `QMP_SEQ_LEN`, `QMP_TRAIN_BATCH`,
`QMP_CONSOLE_DIR`, `QMP_PYTHON`. (Console output goes to `console/`, not `logs/` —
PARAM Shavak forbids writing to any path containing "log".)

## The sweep (≈50 runs)

- **Primary** mamba-130m × 3 seeds: `uniform{2,4,8}`, `posthoc{B3,4,5,6}`,
  `online{B3,4,5,6}` (+ shared 8-bit base) = 36 runs.
- **Second model** mamba2-130m + **transfer** gpt2, 1 seed each ≈ 14 runs.

**Est. time** (mamba-130m, batch 8, seq 1024, 4000 steps): with `mamba_ssm`
~12–18 min/run ⇒ ~5–8 h wall on 2× A4500; without it ×3–5.

## Phase-0 gate

`submit.sh` prints the FP WikiText-2 perplexity for mamba-130m before the sweep
(KL-Lens FP16 reference ≈ 21.6). If it's much higher than ~30, stop and reconsider
the testbed. Report this number before launching the full sweep.

## Outputs

`results/runs/<run_id>.parquet` (one per run, trajectory + final PPL/avg_bits),
`results/audits/<run_id>.jsonl` (controller/audit events),
`results/analysis.parquet`, `results/figures/pareto_*.png`.
