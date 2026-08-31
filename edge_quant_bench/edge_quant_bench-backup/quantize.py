"""Apply quantization scheme to a model. Returns mutated/wrapped model ready for compile."""
import torch
import logging

log = logging.getLogger(__name__)

def apply_scheme(model, scheme: str):
    """
    Apply the requested quantization scheme in-place.
    Returns (model, error_str_or_None).
    FP16/BF16 cast the whole model. torchao schemes mutate nn.Linear.
    """
    scheme = scheme.lower()

    if scheme == "fp32":
        model = model.float()
        return model, None

    if scheme == "fp16":
        model = model.half()
        return model, None

    if scheme == "bf16":
        model = model.bfloat16()
        return model, None

    if scheme == "w8a16":
        from torchao.quantization import quantize_, int8_weight_only
        model = model.half()
        try:
            quantize_(model, int8_weight_only())
        except Exception as e:
            return model, f"quantize_ failed: {e}"
        return model, None

    if scheme == "w4a16":
        # tinygemm needs bf16 activations
        from torchao.quantization import quantize_, int4_weight_only
        model = model.bfloat16()
        try:
            quantize_(model, int4_weight_only(group_size=128))
        except Exception as e:
            return model, f"quantize_ failed: {e}"
        return model, None

    if scheme == "w8a8":
        from torchao.quantization import quantize_, int8_dynamic_activation_int8_weight
        model = model.half()
        try:
            quantize_(model, int8_dynamic_activation_int8_weight())
        except Exception as e:
            return model, f"quantize_ failed: {e}"
        return model, None

    raise ValueError(f"Unknown scheme: {scheme}")
