# experiments/03_phase3_part2.py

import torch
import sys
import statistics
from datasets import load_dataset
from transformers import AutoTokenizer

sys.path.insert(0, '/home/shreya/Coding/optimizer_V2')

device    = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")

# ── load model ──────────────────────────────────────────────
model_final = torch.load('results/checkpoints/phase3_full_model.pt',
                         map_location=device,
                         weights_only=False)   # ← the only fix needed
model_final = model_final.to(device)
print("Model loaded")

# verify layer types
from torchao.quantization.qat.linear import Int8DynActInt4WeightQATLinear
from src.graph.static_converter import StaticScaleLinear

qat_remaining = sum(1 for _, m in model_final.named_modules()
                    if isinstance(m, Int8DynActInt4WeightQATLinear))
static_frozen  = sum(1 for _, m in model_final.named_modules()
                    if isinstance(m, StaticScaleLinear))
print(f"Dynamic QAT layers remaining: {qat_remaining}")
print(f"Static frozen layers:         {static_frozen}")

# ── val data ────────────────────────────────────────────────
dataset_val = load_dataset("glue", "sst2", split="validation")

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

val_batches = encode_and_batch(dataset_val, batch_size=16)
print(f"Val batches: {len(val_batches)}")

# ── accuracy ────────────────────────────────────────────────
model_final.eval()
correct = total = 0
with torch.no_grad():
    for batch in val_batches:
        inputs = {k: v.to(device) for k, v in batch.items() if k != 'labels'}
        labels = batch['labels'].to(device)
        out    = model_final(**inputs)
        correct += (out.logits.argmax(-1) == labels).sum().item()
        total   += labels.size(0)
acc = correct / total * 100
print(f"Phase 3 accuracy: {acc:.2f}%")

# ── latency ─────────────────────────────────────────────────
compiled = torch.compile(model_final, backend="inductor")
model_final.eval()

with torch.no_grad():
    for batch in val_batches[:5]:
        inputs = {k: v.to(device) for k, v in batch.items() if k != 'labels'}
        compiled(**inputs)
torch.cuda.synchronize()

times = []
with torch.no_grad():
    for batch in val_batches[:30]:
        inputs = {k: v.to(device) for k, v in batch.items() if k != 'labels'}
        start = torch.cuda.Event(enable_timing=True)
        end   = torch.cuda.Event(enable_timing=True)
        start.record()
        compiled(**inputs)
        end.record()
        torch.cuda.synchronize()
        times.append(start.elapsed_time(end))

lat_mean = statistics.mean(times)
lat_std  = statistics.stdev(times)

# ── summary ─────────────────────────────────────────────────
print(f"\n{'='*58}")
print(f"  PHASE 3 COMPLETE — FINAL RESULTS")
print(f"{'='*58}")
print(f"  {'Model':<30} {'Latency':>10}  {'Accuracy':>10}")
print(f"  {'-'*50}")
print(f"  {'FP32 baseline':<30} {'12.0ms':>10}  {'—':>10}")
print(f"  {'QAT dynamic':<30} {'34.8ms':>10}  {'87.38%':>10}")
print(f"  {'Phase 2 post-hoc static':<30} {'12.0ms':>10}  {'86.69%':>10}")
print(f"  {'Phase 3 training-aware':<30} {f'{lat_mean:.1f}ms':>10}  {f'{acc:.2f}%':>10}")
print(f"  {'-'*50}")
print(f"  Speedup over QAT dynamic:  {34.8/lat_mean:.2f}x")
print(f"  Accuracy vs QAT dynamic:   {acc - 87.38:+.2f}%")
print(f"  Accuracy vs Phase 2:       {acc - 86.69:+.2f}%")
print(f"  Layers frozen:             {static_frozen}/38")
print(f"{'='*58}")