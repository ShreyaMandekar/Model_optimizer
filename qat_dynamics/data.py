"""GLUE data loaders for SST-2 (primary), MNLI, QNLI.

Returns DataLoaders for train/validation. Also builds a fixed calibration
batch used by the KL sensitivity auditor.
"""

import torch
from torch.utils.data import DataLoader
from datasets import load_dataset
from transformers import AutoTokenizer
from config import (
    TASKS, MAX_SEQ_LEN, TRAIN_BATCH, EVAL_BATCH, KL_CALIB_SAMPLES
)


def get_loaders(task_name: str, model_hf_name: str, seed: int = 42):
    """Return (train_loader, eval_loader, calib_batch) for the given GLUE task."""
    cfg = TASKS[task_name]
    raw = load_dataset("nyu-mll/glue", cfg["glue_name"])

    tokenizer = AutoTokenizer.from_pretrained(model_hf_name)

    def tokenize(batch):
        if "text_col1" in cfg:
            return tokenizer(
                batch[cfg["text_col1"]], batch[cfg["text_col2"]],
                truncation=True, max_length=MAX_SEQ_LEN, padding="max_length"
            )
        return tokenizer(
            batch[cfg["text_col"]], truncation=True,
            max_length=MAX_SEQ_LEN, padding="max_length"
        )

    tokenized = raw.map(tokenize, batched=True,
                        remove_columns=[c for c in raw["train"].column_names
                                        if c not in (cfg.get("label_col", "label"),)])
    tokenized = tokenized.rename_column(cfg.get("label_col", "label"), "labels")
    tokenized.set_format("torch")

    # for MNLI eval use validation_matched
    eval_split = "validation_matched" if task_name == "mnli" else "validation"

    g = torch.Generator(); g.manual_seed(seed)
    train_loader = DataLoader(
        tokenized["train"], batch_size=TRAIN_BATCH, shuffle=True, generator=g,
        drop_last=True
    )
    eval_loader = DataLoader(
        tokenized[eval_split], batch_size=EVAL_BATCH, shuffle=False
    )

    # fixed calibration batch (same every call for reproducibility)
    calib = _build_calib(tokenized["train"], seed=seed)

    return train_loader, eval_loader, calib


def _build_calib(dataset, seed: int = 42):
    """Return a single fixed batch of KL_CALIB_SAMPLES examples."""
    g = torch.Generator(); g.manual_seed(seed)
    indices = torch.randperm(len(dataset), generator=g)[:KL_CALIB_SAMPLES].tolist()
    subset = dataset.select(indices)
    loader = DataLoader(subset, batch_size=KL_CALIB_SAMPLES, shuffle=False)
    return next(iter(loader))


def move_batch(batch: dict, device: torch.device) -> dict:
    return {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
