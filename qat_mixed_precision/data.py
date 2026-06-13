"""WikiText loaders: block-tokenized LM data, a fixed calib batch, perplexity eval."""
from __future__ import annotations
import math
from typing import Optional
import torch
from torch.utils.data import DataLoader, TensorDataset
from datasets import load_dataset

from config import (TRAIN_DATASET, EVAL_DATASET, SEQ_LEN, CALIB_SAMPLES,
                    CALIB_SEQ_LEN, TRAIN_BATCH, EVAL_BATCH)


def _tokenize_blocks(tokenizer, ds_spec, seq_len, max_blocks: Optional[int] = None):
    ds = load_dataset(ds_spec[0], ds_spec[1], split="train" if ds_spec is TRAIN_DATASET else "test")
    text = "\n\n".join(t for t in ds["text"] if t.strip())
    ids = tokenizer(text, return_tensors="pt").input_ids[0]
    n_blocks = ids.numel() // seq_len
    if max_blocks:
        n_blocks = min(n_blocks, max_blocks)
    ids = ids[: n_blocks * seq_len].view(n_blocks, seq_len)
    return ids


def make_train_loader(tokenizer, max_blocks: Optional[int] = None):
    ids = _tokenize_blocks(tokenizer, TRAIN_DATASET, SEQ_LEN, max_blocks)
    return DataLoader(TensorDataset(ids), batch_size=TRAIN_BATCH,
                      shuffle=True, drop_last=True)


def make_eval_ids(tokenizer):
    return _tokenize_blocks(tokenizer, EVAL_DATASET, SEQ_LEN)


def make_calib_batch(tokenizer, device):
    ids = _tokenize_blocks(tokenizer, TRAIN_DATASET, CALIB_SEQ_LEN,
                           max_blocks=CALIB_SAMPLES)
    return {"input_ids": ids.to(device),
            "attention_mask": torch.ones_like(ids).to(device)}


def batch_to_inputs(batch_ids, device):
    ids = batch_ids.to(device)
    return {"input_ids": ids, "attention_mask": torch.ones_like(ids),
            "labels": ids.clone()}


@torch.no_grad()
def eval_perplexity(model, eval_ids, device) -> float:
    model.eval()
    total_loss, total_tok = 0.0, 0
    for i in range(0, eval_ids.size(0), EVAL_BATCH):
        ids = eval_ids[i:i + EVAL_BATCH].to(device)
        out = model(input_ids=ids, attention_mask=torch.ones_like(ids), labels=ids)
        # HF returns mean loss over (seq-1)*batch tokens
        ntok = ids.size(0) * (ids.size(1) - 1)
        total_loss += out.loss.item() * ntok
        total_tok += ntok
    model.train()
    return math.exp(total_loss / max(total_tok, 1))
