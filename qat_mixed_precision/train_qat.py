"""One run = (model, method, tag, seed). Methods: uniform | posthoc_base | posthoc | online.

Crash-safe: writes results/runs/<run_id>.parquet at the end; skip-if-done.
Controller/audit events -> results/audits/<run_id>.jsonl.
"""
from __future__ import annotations
import json
import random
import numpy as np
import pandas as pd
import torch
from transformers import get_cosine_schedule_with_warmup

import config as C
from models import load_model_and_tokenizer
from data import (make_train_loader, make_eval_ids, make_calib_batch,
                  batch_to_inputs, eval_perplexity)
from dyn_precision_linear import get_dp_layers, set_all_bits, avg_bits
from marginal_kl import audit as kl_audit
from precision_controller import PrecisionController, _next_lower
from config import MIN_BITS, MAX_BITS


def set_seed(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)


def make_online_lr_schedule(optimizer, total_steps: int):
    """LR schedule for the online method (schedule co-design, PAPER_VS_CODE_AUDIT.md
    Part 2/5): warmup -> held at peak through anneal + hold/co-adapt -> cosine decay
    to a floor over the final tail. A cosine-to-zero schedule (used by uniform/posthoc)
    would place the largest precision drops where LR has decayed toward zero, leaving
    no capacity to co-adapt.
    """
    import math
    from torch.optim.lr_scheduler import LambdaLR

    warmup_steps = int(C.WARMUP_FRAC * total_steps)
    decay_start_step = int(C.ONLINE_LR_DECAY_START * total_steps)

    def lr_lambda(step):
        if step < warmup_steps:
            return float(step) / max(1, warmup_steps)
        if step < decay_start_step:
            return 1.0
        decay_frac = (step - decay_start_step) / max(1, total_steps - decay_start_step)
        return C.ONLINE_LR_FLOOR + (1.0 - C.ONLINE_LR_FLOOR) * (
            1.0 + math.cos(math.pi * decay_frac)) / 2.0

    return LambdaLR(optimizer, lr_lambda)


def _base_ckpt_path(model, seed):
    return C.CKPTS / f"{model}__seed{seed}__base8.pt"


def _posthoc_assign(model, calib_batch, budget):
    """Greedily drop least-(marginal_down/bit) layers until avg_bits <= budget."""
    set_all_bits(model, MAX_BITS)
    layers = get_dp_layers(model)
    guard = 0
    while avg_bits(model) > budget and guard < len(layers) * 4:
        guard += 1
        au = kl_audit(model, calib_batch, want_up=False, subsample=None)
        au.pop("__base_kl__", None)
        best = None
        for n, r in au.items():
            m = layers[n]
            if m.bit_width is None or m.bit_width <= MIN_BITS:
                continue
            md = r.get("marginal_down")
            if md is None:
                continue
            newb = _next_lower(m.bit_width)
            cost = md / max(m.bit_width - newb, 1)
            if best is None or cost < best[0]:
                best = (cost, n, newb)
        if best is None:
            break
        layers[best[1]].set_bits(best[2])


def train_loop(model, loader, opt, sched, device, total_steps,
               controller=None, log_every=100, eval_ids=None, eval_every=None):
    model.train()
    traj = []
    step = 0
    while step < total_steps:
        for (batch_ids,) in loader:
            if step >= total_steps:
                break
            inp = batch_to_inputs(batch_ids, device)
            out = model(**inp)
            loss = out.loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), C.MAX_GRAD_NORM)
            opt.step(); sched.step(); opt.zero_grad()
            if controller is not None:
                controller.step(step)
            if step % log_every == 0:
                rec = {"step": step, "loss": loss.item(), "avg_bits": avg_bits(model)}
                if eval_every and eval_ids is not None and step > 0 and step % eval_every == 0:
                    rec["ppl"] = eval_perplexity(model, eval_ids, device)
                traj.append(rec)
            step += 1
    return traj


def run(spec: dict, device="cuda"):
    run_id = spec["run_id"]
    out_path = C.RUNS / f"{run_id}.parquet"
    if out_path.exists():
        print(f"[skip] {run_id} (done)")
        return
    print(f"[run ] {run_id}")
    set_seed(spec["seed"])
    device = torch.device(device if torch.cuda.is_available() else "cpu")

    model, tok, wrapped = load_model_and_tokenizer(spec["model"])
    model.to(device)
    loader = make_train_loader(tok)
    eval_ids = make_eval_ids(tok)
    calib = make_calib_batch(tok, device)

    opt = torch.optim.AdamW(model.parameters(), lr=C.LEARNING_RATE,
                            weight_decay=C.WEIGHT_DECAY)
    method = spec["method"]
    controller = None
    events = []

    if method == "uniform":
        set_all_bits(model, spec["bits"])
        steps = C.TOTAL_STEPS
    elif method == "posthoc_base":
        set_all_bits(model, MAX_BITS)
        steps = C.TOTAL_STEPS
    elif method == "posthoc":
        # load shared 8-bit base; fall back to training one
        ckpt = _base_ckpt_path(spec["model"], spec["seed"])
        set_all_bits(model, MAX_BITS)
        if ckpt.exists():
            model.load_state_dict(torch.load(ckpt, map_location=device), strict=False)
            print(f"   loaded base ckpt {ckpt.name}")
        else:
            print("   base ckpt missing -> training base first")
            sched0 = get_cosine_schedule_with_warmup(
                opt, int(C.WARMUP_RATIO * C.TOTAL_STEPS), C.TOTAL_STEPS)
            train_loop(model, loader, opt, sched0, device, C.TOTAL_STEPS)
        _posthoc_assign(model, calib, spec["budget"])
        events.append({"action": "posthoc_assign", "avg_bits": avg_bits(model),
                       "bits": {n: m.bit_width for n, m in get_dp_layers(model).items()}})
        steps = C.POSTHOC_RECOVER_STEPS         # short recovery finetune
        opt = torch.optim.AdamW(model.parameters(), lr=C.LEARNING_RATE * 0.5,
                                weight_decay=C.WEIGHT_DECAY)
    elif method == "online":
        def audit_fn():
            return kl_audit(model, calib)
        controller = PrecisionController(model, spec["budget"], C.TOTAL_STEPS, audit_fn,
                                         recovery=spec.get("recovery", True))
        steps = C.TOTAL_STEPS
    else:
        raise ValueError(method)

    if method == "online":
        # held-then-decay schedule so the optimizer can co-adapt through the
        # anneal instead of doing so at LR~0 (see make_online_lr_schedule)
        sched = make_online_lr_schedule(opt, steps)
    else:
        sched = get_cosine_schedule_with_warmup(
            opt, int(C.WARMUP_RATIO * steps), steps)
    traj = train_loop(model, loader, opt, sched, device, steps,
                      controller=controller, eval_ids=eval_ids,
                      eval_every=max(steps // 4, 1))

    final_ppl = eval_perplexity(model, eval_ids, device)
    final_bits = avg_bits(model)

    # save base checkpoint for posthoc reuse
    if method == "posthoc_base":
        torch.save(model.state_dict(), _base_ckpt_path(spec["model"], spec["seed"]))

    # audits / controller events
    if controller is not None:
        events = controller.events
    if events:
        with open(C.AUDITS / f"{run_id}.jsonl", "w") as f:
            for e in events:
                f.write(json.dumps(e, default=float) + "\n")

    # final result row + trajectory
    df = pd.DataFrame(traj)
    for k in ("run_id", "model", "method", "tag", "seed"):
        df[k] = spec[k]
    df["final_ppl"] = final_ppl
    df["final_avg_bits"] = final_bits
    df["budget"] = spec.get("budget", spec.get("bits"))
    df.to_parquet(out_path)
    print(f"[done] {run_id}  ppl={final_ppl:.3f}  avg_bits={final_bits:.2f}")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_id", required=True)
    args = ap.parse_args()
    spec = next(r for r in C.build_runs() if r["run_id"] == args.run_id)
    run(spec)
