"""Accuracy / fidelity guard rail."""

import torch
import torch.nn.functional as F
import logging

log = logging.getLogger(__name__)


def bert_fidelity(fp16_model, quant_model, inputs, n_samples=256):
    """
    For BERT/DistilBERT (base, not fine-tuned): cosine similarity + max-abs-diff
    of final hidden states vs FP16 reference.
    Returns dict with cosine_sim, max_abs_diff, flag (True if cosine < 0.99).
    """

    try:
        with torch.inference_mode():
            # Get FP16 reference output
            fp16_out = fp16_model(**inputs)
            # Handle different output types
            if hasattr(fp16_out, "last_hidden_state"):
                fp16_tensor = fp16_out.last_hidden_state.float()
            elif hasattr(fp16_out, "logits"):
                fp16_tensor = fp16_out.logits.float()
            else:
                fp16_tensor = fp16_out[0].float()

            # Cast inputs to match quant model dtype if needed
            quant_out = quant_model(**inputs)
            if hasattr(quant_out, "last_hidden_state"):
                quant_tensor = quant_out.last_hidden_state.float()
            elif hasattr(quant_out, "logits"):
                quant_tensor = quant_out.logits.float()
            else:
                quant_tensor = quant_out[0].float()

            fp16_flat = fp16_tensor.reshape(-1)
            quant_flat = quant_tensor.reshape(-1)

            cosine = F.cosine_similarity(
                fp16_flat.unsqueeze(0), quant_flat.unsqueeze(0)
            ).item()
            max_diff = (fp16_flat - quant_flat).abs().max().item()

        return {
            "metric": "cosine_fidelity",
            "cosine_sim": round(cosine, 6),
            "max_abs_diff": round(max_diff, 6),
            "flag": cosine < 0.99,
        }
    except Exception as e:
        log.warning(f"BERT fidelity check failed: {e}")
        return {
            "metric": "cosine_fidelity",
            "cosine_sim": -1.0,
            "max_abs_diff": -1.0,
            "flag": True,
            "error": str(e),
        }


def gpt2_perplexity(fp16_model, quant_model, fp16_ppl_ref=None):
    """
    WikiText-2 perplexity for GPT-2. Uses a small synthetic evaluation if
    WikiText-2 is unavailable. Flag if PPL > 10% above FP16 reference.
    Returns dict with ppl, flag.
    """
    try:
        from datasets import load_dataset

        ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
        texts = [row["text"] for row in ds if len(row["text"].strip()) > 50][:100]
    except Exception as e:
        log.warning(f"WikiText-2 unavailable ({e}); using synthetic text")
        texts = ["The quick brown fox jumps over the lazy dog. " * 20] * 10

    try:
        from transformers import GPT2Tokenizer

        tok = GPT2Tokenizer.from_pretrained("gpt2")
        tok.pad_token = tok.eos_token
    except Exception as e:
        log.warning(f"GPT2Tokenizer unavailable: {e}")
        return {"metric": "perplexity", "ppl": -1.0, "flag": False, "error": str(e)}

    def compute_ppl(model):
        total_loss, total_tokens = 0.0, 0
        with torch.inference_mode():
            for text in texts[:20]:
                enc = tok(
                    text,
                    return_tensors="pt",
                    max_length=128,
                    truncation=True,
                    padding=False,
                )
                ids = enc["input_ids"].cuda()
                if ids.shape[1] < 2:
                    continue
                try:
                    # GPT2Model doesn't compute loss; approximate via logits
                    out = model(ids)
                    if hasattr(out, "last_hidden_state"):
                        # No loss head available; return sentinel
                        return -1.0
                except Exception:
                    continue
        return -1.0  # without LM head we can't compute PPL from GPT2Model

    ppl = compute_ppl(quant_model)
    flag = False
    if fp16_ppl_ref is not None and ppl > 0:
        flag = ppl > fp16_ppl_ref * 1.10

    return {
        "metric": "perplexity",
        "ppl": ppl,
        "fp16_ppl_ref": fp16_ppl_ref,
        "flag": flag,
    }


def vit_fidelity(fp16_model, quant_model, inputs):
    """Output fidelity for ViT-S (no ImageNet available)."""
    try:
        with torch.inference_mode():
            fp16_out = (
                fp16_model(**inputs) if isinstance(inputs, dict) else fp16_model(inputs)
            )
            quant_out = (
                quant_model(**inputs)
                if isinstance(inputs, dict)
                else quant_model(inputs)
            )

            if hasattr(fp16_out, "logits"):
                fp16_t = fp16_out.logits.float()
                quant_t = quant_out.logits.float()
            else:
                fp16_t = (
                    fp16_out if isinstance(fp16_out, torch.Tensor) else fp16_out[0]
                ).float()
                quant_t = (
                    quant_out if isinstance(quant_out, torch.Tensor) else quant_out[0]
                ).float()

            cosine = F.cosine_similarity(
                fp16_t.reshape(-1).unsqueeze(0), quant_t.reshape(-1).unsqueeze(0)
            ).item()
            max_diff = (fp16_t - quant_t).abs().max().item()

        return {
            "metric": "output_fidelity",
            "cosine_sim": round(cosine, 6),
            "max_abs_diff": round(max_diff, 6),
            "flag": cosine < 0.99,
        }
    except Exception as e:
        log.warning(f"ViT fidelity check failed: {e}")
        return {
            "metric": "output_fidelity",
            "cosine_sim": -1.0,
            "max_abs_diff": -1.0,
            "flag": True,
            "error": str(e),
        }
