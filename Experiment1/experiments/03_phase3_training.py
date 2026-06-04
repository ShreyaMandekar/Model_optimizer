# experiments/03_phase3_training.py
import torch._dynamo
torch._dynamo.config.cache_size_limit = 64
import torch
import torch.nn as nn
import copy
import json
import statistics
import os
import matplotlib.pyplot as plt
from datasets import load_dataset
from transformers import (AutoTokenizer, AutoModelForSequenceClassification,
                          DistilBertConfig, DistilBertForSequenceClassification)
from torchao.quantization.qat.linear import Int8DynActInt4WeightQATLinear

import sys
sys.path.insert(0, '/home/shreya/Coding/optimizer_V2')
from Experiment1.src.training.trainer import FusionAwareTrainer

# ── device ─────────────────────────────────────────────────
device    = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")

# ── QAT application — exact same function as your notebook ─
def apply_qat_to_linear(model):
    for name, module in model.named_children():
        if isinstance(module, torch.nn.Linear):
            qat_linear = Int8DynActInt4WeightQATLinear(
                module.in_features,
                module.out_features,
                bias=False,
                groupsize=32,
            )
            with torch.no_grad():
                qat_linear.weight.copy_(module.weight)
            if module.bias is not None:
                qat_linear.bias = nn.Parameter(module.bias.clone())
            setattr(model, name, qat_linear)
        else:
            apply_qat_to_linear(module)
    return model

# ── data ───────────────────────────────────────────────────
dataset_train = load_dataset("glue", "sst2", split="train[:5000]")
dataset_val   = load_dataset("glue", "sst2", split="validation")

def encode_and_batch(dataset, batch_size=16):
    sentences = list(dataset['sentence'])
    labels    = list(dataset['label'])
    enc  = tokenizer(sentences, padding='max_length',
                     truncation=True, max_length=64)
    ids   = torch.tensor(enc['input_ids'])
    masks = torch.tensor(enc['attention_mask'])
    labs  = torch.tensor(labels)
    batches = []
    for i in range(0, len(ids) - batch_size, batch_size):
        batches.append({
            'input_ids':      ids[i:i+batch_size],
            'attention_mask': masks[i:i+batch_size],
            'labels':         labs[i:i+batch_size]
        })
    return batches

train_batches = encode_and_batch(dataset_train, batch_size=16)
val_batches   = encode_and_batch(dataset_val,   batch_size=16)
print(f"Train: {len(train_batches)} batches  Val: {len(val_batches)} batches")

# ── model loading ───────────────────────────────────────────
# Step 1: build plain model
config            = DistilBertConfig()
config.num_labels = 2
base_model        = DistilBertForSequenceClassification(config).to(device)

# Step 2: load fine-tuned weights
ckpt = torch.load('results/checkpoints/qat_finetuned.pt', map_location=device)
base_model.load_state_dict(ckpt, strict=True)
print("Checkpoint loaded")


# Step 3: apply QAT using the same function as your notebook
base_model = apply_qat_to_linear(base_model)
base_model = base_model.to(device)

# Step 4: verify QAT layers present
qat_count = sum(
    1 for _, m in base_model.named_modules()
    if isinstance(m, Int8DynActInt4WeightQATLinear)
)
print(f"QAT layers: {qat_count}")  # must be 38, not 0

# Step 5: verify accuracy
base_model.eval()
correct = total = 0
with torch.no_grad():
    for batch in val_batches:
        inputs = {k: v.to(device) for k, v in batch.items() if k != 'labels'}
        labels = batch['labels'].to(device)
        out    = base_model(**inputs)
        correct += (out.logits.argmax(-1) == labels).sum().item()
        total   += labels.size(0)
print(f"Accuracy before Phase 3: {correct/total*100:.2f}%")  # should be 87.38%

model_p3 = copy.deepcopy(base_model)

# ── train ───────────────────────────────────────────────────
trainer = FusionAwareTrainer(
    model            = model_p3,
    train_batches    = train_batches,
    val_batches      = val_batches,
    device           = device,
    lr               = 1e-5,
    cv_threshold     = 5.0,
    patience = 50,
    freeze_tier_size = 3,
    adaptation_steps = 150
)

model_final = trainer.train(epochs=5)

# in 03_phase3_part2.py, after loading model_final
# run Phase 2 calibration on the 9 remaining dynamic layers

from Experiment1.src.graph.boundary_detector import ActivationStabilityDetector
from Experiment1.src.graph.static_converter import convert_stable_layers
import copy

detector = ActivationStabilityDetector(cv_threshold_static=5.0)
detector.attach_hooks(model_final)

model_final.eval()
with torch.no_grad():
    for batch in val_batches[:25]:
        inputs = {k: v.to(device) for k, v in batch.items() if k != 'labels'}
        model_final(**inputs)

detector.remove_hooks()
labels_remaining = detector.compute_labels()

# only labels for the 9 still-dynamic layers will be populated
model_combined, report = convert_stable_layers(
    copy.deepcopy(model_final), labels_remaining, include_borderline=False
)

# measure
combined_static = sum(1 for _, m in model_combined.named_modules()
                      if isinstance(m, StaticScaleLinear))
print(f"Combined frozen layers: {combined_static}/38")

# accuracy + latency

# ── latency ─────────────────────────────────────────────────
def measure_latency(model, batches, warmup=5, runs=20):
    import torch  # explicit import inside function avoids scope issues
    import statistics
    
    _device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    compiled = torch.compile(model, backend="inductor")
    model.eval()
    
    with torch.no_grad():
        for batch in batches[:warmup]:
            inputs = {k: v.to(_device) for k, v in batch.items()
                      if k != 'labels'}
            compiled(**inputs)
    
    times = []
    with torch.no_grad():
        for batch in batches[:runs]:
            inputs = {k: v.to(_device) for k, v in batch.items()
                      if k != 'labels'}
            start = torch.cuda.Event(enable_timing=True)
            end   = torch.cuda.Event(enable_timing=True)
            start.record()
            compiled(**inputs)
            end.record()
            torch.cuda.synchronize()
            times.append(start.elapsed_time(end))
    
    return statistics.mean(times), statistics.stdev(times)

# add this right after trainer.train() completes, before measure_latency
torch.save(model_final, 'results/checkpoints/phase3_full_model.pt')
print("Full model saved")