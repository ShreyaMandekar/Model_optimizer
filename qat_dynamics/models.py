"""Model factory: build + QAT-prepare + wrap with DynStatQATLinear.

Keeps Experiment1's trainer.py out of import path.
"""

import torch
import torch.nn as nn
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from qat_linear import DynStatQATLinear, wrap_model
from config import MODELS, TASKS


def build_model(model_key: str, task_name: str, seed: int = 42):
    """Return (model, tokenizer) QAT-prepared and wrapped with DynStatQATLinear."""
    torch.manual_seed(seed)
    cfg      = MODELS[model_key]
    task_cfg = TASKS[task_name]

    model = AutoModelForSequenceClassification.from_pretrained(
        cfg["hf_name"], num_labels=task_cfg["num_labels"]
    )
    tokenizer = AutoTokenizer.from_pretrained(cfg["hf_name"])

    # QAT prepare: directly replace nn.Linear → DynStatQATLinear (handles bias layers).
    # We skip torchao's stock quantizer.prepare() because it filters out bias layers,
    # which blocks BERT/DistilBERT. Our wrap_model does the same replacement with
    # bias stored as _extra_bias.
    model = wrap_model(model)

    return model, tokenizer


def count_qat_layers(model: nn.Module) -> int:
    return sum(1 for m in model.modules() if isinstance(m, DynStatQATLinear))


def get_qat_layers(model: nn.Module):
    """Yield (name, module) for all DynStatQATLinear in the model."""
    for name, module in model.named_modules():
        if isinstance(module, DynStatQATLinear):
            yield name, module
