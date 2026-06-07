"""Single QAT training run.

Usage:
    python train_qat.py --model distilbert --task sst2 --metric kl --strategy s3_reactive --seed 42

Writes: results/runs/{model}__{task}__{metric}__{strategy}__seed{seed}.parquet
"""

import argparse
import json
import time
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from config import (
    NUM_EPOCHS, LEARNING_RATE, WEIGHT_DECAY, WARMUP_RATIO,
    LOG_FREQ_EARLY, LOG_FREQ_LATE, EARLY_FRACTION, AUDIT_EVERY,
    RUNS, AUDITS, KL_CALIB_SAMPLES,
)
from models import build_model, get_qat_layers
from data import get_loaders, move_batch
from tracker import OnlineStabilityTracker
from kl_sensitivity import compute_layer_kl
from controller import Controller


def train_run(model_key: str, task: str, metric: str, strategy: str, seed: int,
              device_str: str = "cuda") -> dict:
    device = torch.device(device_str if torch.cuda.is_available() else "cpu")
    run_id = f"{model_key}__{task}__{metric}__{strategy}__seed{seed}"
    out_path = RUNS / f"{run_id}.parquet"
    audit_path = AUDITS / f"{run_id}_audits.jsonl"

    if out_path.exists():
        print(f"[skip] {run_id} already done")
        tbl = pq.read_table(out_path)
        return {"run_id": run_id, "skipped": True}

    print(f"\n{'='*70}")
    print(f"RUN: {run_id}")
    print(f"{'='*70}")

    model, tokenizer = build_model(model_key, task, seed=seed)
    model = model.to(device)

    train_loader, eval_loader, calib_batch = get_loaders(task, model.config._name_or_path, seed=seed)
    calib_batch_dev = move_batch(calib_batch, device)

    n_steps_per_epoch = len(train_loader)
    total_steps = NUM_EPOCHS * n_steps_per_epoch
    warmup_steps = int(WARMUP_RATIO * total_steps)

    optimizer = AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = get_linear_schedule_with_warmup(optimizer, warmup_steps, total_steps)

    # tracker
    tracker = OnlineStabilityTracker()
    tracker.attach_hooks(model)

    # kl_fn closure
    def kl_fn(layer_names):
        return compute_layer_kl(model, calib_batch_dev, layer_names=layer_names, device=device)

    # val_fn closure
    def val_fn():
        return _eval_loss(model, eval_loader, device)

    ctrl = Controller(
        model=model, metric=metric, strategy=strategy,
        total_steps=total_steps,
        tracker=tracker, kl_fn=kl_fn, val_fn=val_fn,
    )

    # calib_forward for final recalibrate
    def calib_forward():
        model.eval()
        with torch.no_grad():
            fwd = {k: v for k, v in calib_batch_dev.items() if k != "labels"}
            model(**fwd)
        model.train()

    log_rows = []
    audit_rows = []
    global_step = 0

    for epoch in range(NUM_EPOCHS):
        model.train()
        epoch_loss = 0.0
        for batch in train_loader:
            batch = move_batch(batch, device)
            global_step += 1

            outputs = model(**batch)
            loss = outputs.loss
            loss.backward()

            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

            # tracker step
            tracker_states = tracker.step(frozen_layers=set(ctrl.frozen_names()))

            # controller step
            ctrl.step(tracker_states=tracker_states)

            epoch_loss += loss.item()

            # --- logging ---
            frac = global_step / total_steps
            freq = LOG_FREQ_EARLY if frac < EARLY_FRACTION else LOG_FREQ_LATE
            if global_step % freq == 0:
                row = {"step": global_step, "epoch": epoch, "loss": loss.item(),
                       "frozen_fraction": ctrl.frozen_fraction()}
                # per-layer scalars
                for name, st in tracker_states.items():
                    row[f"cv__{name}"]    = st.get("cv", None)
                    row[f"scale__{name}"] = st.get("ema_mean", None)
                    row[f"state__{name}"] = ctrl.state[name].value
                log_rows.append(row)

            # KL audit log every AUDIT_EVERY steps
            if global_step % AUDIT_EVERY == 0 and metric == "kl":
                kl = kl_fn(ctrl.frozen_names() or list({n for n, _ in get_qat_layers(model)}))
                for name, v in kl.items():
                    audit_rows.append({"step": global_step, "layer": name, "kl": v})

        # end of epoch: eval
        val_acc, val_loss = _eval(model, eval_loader, device)
        ctrl.check_regression(val_loss)
        print(f"  Epoch {epoch+1}/{NUM_EPOCHS} | "
              f"train_loss={epoch_loss/len(train_loader):.4f} | "
              f"val_acc={val_acc:.4f} | frozen={ctrl.frozen_fraction():.2f}")

    # --- end of training ---
    # s2/s3/s4 finalize
    ctrl.final_recalibrate(calib_forward)
    ctrl.finalize_soft()

    # final eval
    final_acc, final_loss = _eval(model, eval_loader, device)
    final_frozen_frac = ctrl.frozen_fraction()
    print(f"  FINAL: acc={final_acc:.4f} frozen={final_frozen_frac:.2f}")

    # --- save parquet ---
    summary = {
        "run_id":       run_id,
        "model":        model_key,
        "task":         task,
        "metric":       metric,
        "strategy":     strategy,
        "seed":         seed,
        "final_acc":    final_acc,
        "final_loss":   final_loss,
        "frozen_frac":  final_frozen_frac,
        "n_layers":     ctrl.n_layers,
        "n_frozen":     len(ctrl.frozen_names()),
        "total_steps":  total_steps,
        "controller_events": json.dumps(ctrl.events),
        "state_summary": json.dumps(ctrl.state_summary()),
    }
    # append summary row to log
    log_rows.append(summary)

    df = pd.DataFrame(log_rows)
    pq.write_table(pa.Table.from_pandas(df), out_path)
    print(f"  Saved: {out_path}")

    if audit_rows:
        with open(audit_path, "w") as f:
            for r in audit_rows:
                f.write(json.dumps(r) + "\n")

    return summary


def _eval(model, loader, device):
    model.eval()
    correct = total = 0
    total_loss = 0.0
    with torch.no_grad():
        for batch in loader:
            batch = move_batch(batch, device)
            out = model(**batch)
            preds = out.logits.argmax(-1)
            labels = batch["labels"]
            correct += (preds == labels).sum().item()
            total   += labels.size(0)
            total_loss += out.loss.item()
    model.train()
    return correct / max(total, 1), total_loss / max(len(loader), 1)


def _eval_loss(model, loader, device):
    _, loss = _eval(model, loader, device)
    return loss


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model",    default="distilbert")
    parser.add_argument("--task",     default="sst2")
    parser.add_argument("--metric",   default="kl", choices=["cv", "kl"])
    parser.add_argument("--strategy", default="s3_reactive",
                        choices=["s1_oneway", "s2_recal", "s3_reactive", "s4_soft"])
    parser.add_argument("--seed",     type=int, default=42)
    args = parser.parse_args()

    train_run(args.model, args.task, args.metric, args.strategy, args.seed)


if __name__ == "__main__":
    main()
