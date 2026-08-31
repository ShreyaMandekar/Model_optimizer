# experiments/05_generalization.py

import torch
import torch._dynamo
import sys
import copy
import statistics
import json
import os
from datasets import load_dataset
from transformers import AutoTokenizer, BertForSequenceClassification, BertConfig
from torchao.quantization.qat.linear import Int8DynActInt4WeightQATLinear

sys.path.insert(0, '/home/shreya/Coding/optimizer_V2')
from Experiment1.src.graph.boundary_detector import ActivationStabilityDetector
from Experiment1.src.graph.static_converter import StaticScaleLinear, convert_stable_layers

torch.cuda.empty_cache()
torch.set_float32_matmul_precision('high')
torch._dynamo.config.cache_size_limit = 64
os.makedirs('results/checkpoints', exist_ok=True)
os.makedirs('results/profiles',    exist_ok=True)

device    = torch.device('cuda')
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

# ── apply QAT — same function as your main experiments ───────
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
                qat_linear.bias = torch.nn.Parameter(module.bias.clone())
            setattr(model, name, qat_linear)
        else:
            apply_qat_to_linear(module)
    return model

# ── data — more training data than before ────────────────────
dataset_train = load_dataset("glue", "sst2", split="train[:10000]")  # 2x more
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

# ── build BERT-base QAT ───────────────────────────────────────
bert_model = BertForSequenceClassification.from_pretrained(
    "bert-base-uncased", num_labels=2
).to(device)

bert_qat = apply_qat_to_linear(bert_model)
bert_qat = bert_qat.to(device)

qat_count = sum(1 for _, m in bert_qat.named_modules()
                if isinstance(m, Int8DynActInt4WeightQATLinear))
print(f"BERT-base QAT layers: {qat_count}")

# ── fine-tune properly ────────────────────────────────────────
optimizer = torch.optim.AdamW(bert_qat.parameters(), lr=2e-5,
                               weight_decay=0.01)
scheduler = torch.optim.lr_scheduler.LinearLR(
    optimizer, start_factor=1.0, end_factor=0.0,
    total_iters=5 * len(train_batches)
)
criterion = torch.nn.CrossEntropyLoss()

best_acc   = 0.0
best_epoch = 0

print("\nTraining BERT-base QAT on SST-2...")
for epoch in range(5):
    bert_qat.train()
    losses = []
    for batch in train_batches:
        inputs = {k: v.to(device) for k, v in batch.items()
                  if k != 'labels'}
        labels = batch['labels'].to(device)
        optimizer.zero_grad()
        out  = bert_qat(**inputs)
        loss = criterion(out.logits, labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(bert_qat.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        losses.append(loss.item())

    # validate
    bert_qat.eval()
    correct = total = 0
    with torch.no_grad():
        for batch in val_batches:
            inputs = {k: v.to(device) for k, v in batch.items()
                      if k != 'labels'}
            labels = batch['labels'].to(device)
            out    = bert_qat(**inputs)
            correct += (out.logits.argmax(-1) == labels).sum().item()
            total   += labels.size(0)
    acc = correct / total * 100
    print(f"Epoch {epoch+1}/5  loss={statistics.mean(losses):.4f}  "
          f"val_acc={acc:.2f}%")

    if acc > best_acc:
        best_acc   = acc
        best_epoch = epoch + 1
        torch.save(bert_qat.state_dict(),
                   'results/checkpoints/bert_base_qat_best.pt')

print(f"\nBest accuracy: {best_acc:.2f}% at epoch {best_epoch}")
# target: 90%+ — if not reached, stop here and paste output

# ── stop if accuracy too low ──────────────────────────────────
if best_acc < 88.0:
    print(f"\nAccuracy {best_acc:.2f}% is below 88% threshold.")
    print("Do not proceed with conversion — paste this output.")
    exit()

# ── load best checkpoint ──────────────────────────────────────
ckpt = torch.load('results/checkpoints/bert_base_qat_best.pt',
                  map_location=device)
bert_qat.load_state_dict(ckpt)
print(f"\nLoaded best checkpoint ({best_acc:.2f}%)")

# ── measure QAT dynamic latency ───────────────────────────────
def measure_latency(model, warmup=10, runs=50):
    compiled = torch.compile(model, backend="inductor")
    model.eval()
    with torch.no_grad():
        for batch in val_batches[:warmup]:
            inputs = {k: v.to(device) for k, v in batch.items()
                      if k != 'labels'}
            compiled(**inputs)
    torch.cuda.synchronize()
    times = []
    with torch.no_grad():
        for batch in val_batches[:runs]:
            inputs = {k: v.to(device) for k, v in batch.items()
                      if k != 'labels'}
            start = torch.cuda.Event(enable_timing=True)
            end   = torch.cuda.Event(enable_timing=True)
            start.record()
            compiled(**inputs)
            end.record()
            torch.cuda.synchronize()
            times.append(start.elapsed_time(end))
    return statistics.mean(times), statistics.stdev(times)

def measure_accuracy(model):
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for batch in val_batches:
            inputs = {k: v.to(device) for k, v in batch.items()
                      if k != 'labels'}
            labels = batch['labels'].to(device)
            out    = model(**inputs)
            correct += (out.logits.argmax(-1) == labels).sum().item()
            total   += labels.size(0)
    return correct / total * 100

print("\nMeasuring BERT-base QAT dynamic latency...")
lat_qat, std_qat = measure_latency(bert_qat)
acc_qat          = measure_accuracy(bert_qat)
print(f"QAT dynamic: {lat_qat:.1f}ms ± {std_qat:.1f}ms  acc={acc_qat:.2f}%")

# ── stability analysis ────────────────────────────────────────
print("\nRunning stability analysis on BERT-base...")
detector = ActivationStabilityDetector(cv_threshold_static=5.0)
detector.attach_hooks(bert_qat)

bert_qat.eval()
with torch.no_grad():
    for batch in val_batches:
        inputs = {k: v.to(device) for k, v in batch.items()
                  if k != 'labels'}
        bert_qat(**inputs)

detector.remove_hooks()
labels_bert = detector.compute_labels()
detector.summary(labels_bert)

# show CV distribution
static_count = sum(1 for v in labels_bert.values()
                   if v['verdict'] == 'STATIC_OK')
border_count = sum(1 for v in labels_bert.values()
                   if v['verdict'] == 'BORDERLINE')
print(f"\nBERT-base stability breakdown:")
print(f"  STATIC OK  (cv < 5%):  {static_count}/{qat_count}")
print(f"  BORDERLINE (5-15%):    {border_count}/{qat_count}")

# ── convert stable layers ─────────────────────────────────────
bert_converted, report = convert_stable_layers(
    copy.deepcopy(bert_qat), labels_bert, include_borderline=False
)
frozen = len(report['converted'])
print(f"\nConverted {frozen}/{qat_count} layers to static scale")

# ── measure converted model ───────────────────────────────────
print("\nMeasuring BERT-base static converted latency...")
lat_conv, std_conv = measure_latency(bert_converted)
acc_conv           = measure_accuracy(bert_converted)
print(f"Static converted: {lat_conv:.1f}ms ± {std_conv:.1f}ms  "
      f"acc={acc_conv:.2f}%")

# ── compare with FP32 ─────────────────────────────────────────
bert_fp32 = BertForSequenceClassification.from_pretrained(
    "bert-base-uncased", num_labels=2
).to(device)
# load same weights but without QAT
from transformers import DistilBertConfig
config_b = bert_fp32.config
ckpt2    = torch.load('results/checkpoints/bert_base_qat_best.pt',
                      map_location=device)
# load only matching keys
bert_fp32_state = bert_fp32.state_dict()
for k in bert_fp32_state:
    if k in ckpt2:
        bert_fp32_state[k] = ckpt2[k]
bert_fp32.load_state_dict(bert_fp32_state)

lat_fp32, std_fp32 = measure_latency(bert_fp32)
print(f"FP32 baseline: {lat_fp32:.1f}ms ± {std_fp32:.1f}ms")

# ── save results ──────────────────────────────────────────────
results = {
    'model':           'bert-base-uncased',
    'task':            'sst2',
    'qat_layers':      qat_count,
    'static_ok':       static_count,
    'borderline':      border_count,
    'fp32_latency':    lat_fp32,
    'qat_latency':     lat_qat,
    'converted_latency': lat_conv,
    'qat_accuracy':    acc_qat,
    'converted_accuracy': acc_conv,
    'accuracy_drop':   acc_qat - acc_conv,
    'speedup':         lat_qat / lat_conv,
    'layers_frozen':   frozen
}
with open('results/profiles/generalization_bert_base.json', 'w') as f:
    json.dump(results, f, indent=2)

# ── final table ───────────────────────────────────────────────
print(f"\n{'='*62}")
print(f"  BERT-BASE GENERALIZATION RESULTS")
print(f"{'='*62}")
print(f"  {'Model':<35} {'Latency':>9}  {'Accuracy':>10}")
print(f"  {'-'*57}")
print(f"  {'BERT-base FP32':<35} {f'{lat_fp32:.1f}ms':>9}  {'—':>10}")
print(f"  {'BERT-base QAT dynamic':<35} {f'{lat_qat:.1f}ms':>9}  "
      f"{f'{acc_qat:.2f}%':>10}")
print(f"  {'BERT-base static converted':<35} {f'{lat_conv:.1f}ms':>9}  "
      f"{f'{acc_conv:.2f}%':>10}")
print(f"  {'-'*57}")
print(f"  Speedup:            {lat_qat/lat_conv:.2f}x")
print(f"  Accuracy cost:      {acc_qat - acc_conv:.2f}%")
print(f"  Layers frozen:      {frozen}/{qat_count}")
print(f"  Static OK rate:     {static_count/qat_count*100:.0f}%")
print(f"{'='*62}")
print("\nSaved: results/profiles/generalization_bert_base.json")