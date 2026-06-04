"""Single source of truth for the sweep matrix and constants."""

MODELS_PHASE1 = ["distilbert-base", "bert-base", "gpt2", "vit-s"]
MODELS_PHASE2 = ["resnet50"]  # conv-heavy; ONNX/TRT only

SCHEMES_PHASE1 = ["fp32", "fp16", "bf16", "w8a16", "w4a16", "w8a8"]
SCHEMES_PHASE2 = ["fp32", "fp16", "int8_ort"]  # ORT-side

BACKENDS_PHASE1 = ["eager", "compile"]
BACKENDS_PHASE2 = ["ort"]

BATCHES = [1, 4, 16, 64]
SEQ_LEN_DEFAULT = 128  # NLP fixed seq for Phase 1
SEQ_LENS_PHASE3 = [32, 128, 512]  # seq-len sensitivity sweep

IMAGE_SIZE = 224

# Timing
WARMUP_ITERS = 30
MEASURE_BLOCKS = 3
MEASURE_ITERS_PER_BLOCK = 100

# Thermal gate
COOLDOWN_TARGET_TEMP_C = 55
COOLDOWN_MAX_WAIT_S = 60

# Accuracy flags
GPT2_PPL_THRESHOLD = 1.10   # flag if PPL > 10% above FP16
BERT_COSINE_THRESHOLD = 0.99  # flag if cosine < 0.99

# Deterministic: theoretical speedup models
# Memory-bound (batch=1): weight_bytes(FP16) / weight_bytes(scheme)
# Compute-bound (large batch): bit_width(FP16) / bit_width(scheme)
THEORETICAL_SPEEDUP = {
    "fp32":  {"memory_bound": 0.5,  "compute_bound": 0.5},   # FP32 is worse than FP16
    "fp16":  {"memory_bound": 1.0,  "compute_bound": 1.0},   # baseline
    "bf16":  {"memory_bound": 1.0,  "compute_bound": 1.0},
    "w8a16": {"memory_bound": 2.0,  "compute_bound": 1.0},   # half weight bytes
    "w4a16": {"memory_bound": 4.0,  "compute_bound": 1.0},   # quarter weight bytes
    "w8a8":  {"memory_bound": 2.0,  "compute_bound": 2.0},   # half weights+acts
}

RESULTS_DIR = "results"
RAW_DIR = "results/raw"
NCU_DIR = "results/ncu"
FIGURES_DIR = "results/figures"
LOGS_DIR = "logs"
