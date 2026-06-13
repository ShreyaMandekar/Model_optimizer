"""Model loading + wrapping the projection Linears with DynPrecisionLinear."""
from __future__ import annotations
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from config import MODELS
from dyn_precision_linear import wrap_model, get_dp_layers


def load_model_and_tokenizer(model_key: str):
    cfg = MODELS[model_key]
    tok = AutoTokenizer.from_pretrained(cfg["hf_id"])
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token or tok.sep_token or tok.unk_token
    model = AutoModelForCausalLM.from_pretrained(cfg["hf_id"], dtype=torch.float32)
    model, wrapped = wrap_model(model, cfg["target_suffixes"])
    if not wrapped:
        raise RuntimeError(
            f"No layers wrapped for {model_key}; check target_suffixes={cfg['target_suffixes']}")
    return model, tok, wrapped


def wrap_summary(model_key: str) -> dict:
    model, tok, wrapped = load_model_and_tokenizer(model_key)
    layers = get_dp_layers(model)
    nparams = sum(m.num_weight_params() for m in layers.values())
    return {"model": model_key, "n_wrapped": len(wrapped),
            "wrapped_params": nparams,
            "example_names": wrapped[:6]}


if __name__ == "__main__":
    import sys
    key = sys.argv[1] if len(sys.argv) > 1 else "mamba-130m"
    import json
    print(json.dumps(wrap_summary(key), indent=2))
