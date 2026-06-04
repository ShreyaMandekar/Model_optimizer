"""Model registry: name -> {model, sample_input_fn, kind, eval_fn}."""
import torch

_cache = {}

def _load_distilbert():
    from transformers import DistilBertModel, DistilBertConfig
    cfg = DistilBertConfig()
    m = DistilBertModel(cfg).eval().cuda()
    return m

def _load_bert():
    from transformers import BertModel, BertConfig
    cfg = BertConfig()
    m = BertModel(cfg).eval().cuda()
    return m

def _load_gpt2():
    from transformers import GPT2Model, GPT2Config
    cfg = GPT2Config()
    m = GPT2Model(cfg).eval().cuda()
    return m

def _load_vit_s():
    import timm
    m = timm.create_model("vit_small_patch16_224", pretrained=False).eval().cuda()
    return m

def _load_resnet50():
    import timm
    m = timm.create_model("resnet50", pretrained=False).eval().cuda()
    return m

_LOADERS = {
    "distilbert-base": _load_distilbert,
    "bert-base": _load_bert,
    "gpt2": _load_gpt2,
    "vit-s": _load_vit_s,
    "resnet50": _load_resnet50,
}

_KINDS = {
    "distilbert-base": "encoder",
    "bert-base": "encoder",
    "gpt2": "decoder",
    "vit-s": "vit",
    "resnet50": "cnn",
}

def _nlp_input(batch, seq):
    ids = torch.randint(0, 30000, (batch, seq), dtype=torch.long, device="cuda")
    mask = torch.ones(batch, seq, dtype=torch.long, device="cuda")
    return {"input_ids": ids, "attention_mask": mask}

def _vit_input(batch, seq=None):
    # timm VisionTransformer.forward(x) takes a plain tensor, not pixel_values
    return torch.randn(batch, 3, 224, 224, device="cuda")

def _cnn_input(batch, seq=None):
    return torch.randn(batch, 3, 224, 224, device="cuda")

_INPUT_FNS = {
    "distilbert-base": _nlp_input,
    "bert-base": _nlp_input,
    "gpt2": _nlp_input,
    "vit-s": _vit_input,
    "resnet50": _cnn_input,
}

def get_model(name):
    """Return a fresh (uncached) model loaded on CUDA."""
    if name not in _LOADERS:
        raise ValueError(f"Unknown model: {name}")
    return _LOADERS[name]()

def get_sample_input(name, batch, seq=128):
    fn = _INPUT_FNS[name]
    if name in ("vit-s", "resnet50"):
        return fn(batch)
    return fn(batch, seq)

def get_kind(name):
    return _KINDS[name]

def run_model(model, inputs):
    """Unified forward pass returning last-layer output tensor."""
    if isinstance(inputs, dict):
        return model(**inputs)
    return model(inputs)
