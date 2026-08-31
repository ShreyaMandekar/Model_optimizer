"""Budgeted closed-loop precision controller (the `online` method).

Schedule over total_steps (param-weighted average weight-bits):
  warmup [0, WARMUP_FRAC)         : all 8-bit, gather stats, no changes
  anneal [WARMUP_FRAC, HOLD_FRAC) : target descends 8 -> B*; greedily drop layers
  hold   [HOLD_FRAC, 1]           : target pinned at B*; weights co-adapt; recovery on

Decisions use the marginal-KL audit (forward-only):
  drop  : greedily drop the layer with smallest marginal_down PER BIT until avg<=target
  raise : recover any layer with marginal_up > p<RECOVER_PCTL> of the audit's marginals

Anti-thrashing: per-layer cooldown after any change; RAISE_CAP bounds raises
(guarantees termination); drop/raise asymmetry (raise needs a high percentile).
"""
from __future__ import annotations
from typing import Callable, Dict, List, Optional
import numpy as np

from config import (
    BIT_SET, MAX_BITS, MIN_BITS, WARMUP_FRAC, HOLD_FRAC,
    ANNEAL_AUDIT_EVERY, HOLD_AUDIT_EVERY,
    RECOVER_PCTL, COOLDOWN_STEPS, RAISE_CAP,
)
from dyn_precision_linear import get_dp_layers, avg_bits


def _next_lower(b: int) -> int:
    lows = [x for x in BIT_SET if x < b]
    return max(lows) if lows else b


def _next_higher(b: int) -> int:
    highs = [x for x in BIT_SET if x > b]
    return min(highs) if highs else b


class PrecisionController:
    def __init__(self, model, budget_target: float, total_steps: int,
                 audit_fn: Callable, recovery: bool = True):
        self.model = model
        self.layers = get_dp_layers(model)
        self.budget = float(budget_target)        # B*
        self.total_steps = total_steps
        self.audit_fn = audit_fn                   # callable() -> {layer: {marginal_*}}
        self.recovery = recovery

        self.nparams = {n: m.num_weight_params() for n, m in self.layers.items()}
        self.cooldown: Dict[str, int] = {n: 0 for n in self.layers}
        self.raise_count: Dict[str, int] = {n: 0 for n in self.layers}
        self.events: List[dict] = []

        # start everyone at 8-bit
        for m in self.layers.values():
            m.set_bits(MAX_BITS)

        self._warm = int(WARMUP_FRAC * total_steps)
        self._hold = int(HOLD_FRAC * total_steps)

    # -- schedule ----------------------------------------------------------
    def target_bits(self, step: int) -> float:
        if step < self._warm:
            return float(MAX_BITS)
        if step >= self._hold:
            return self.budget
        frac = (step - self._warm) / max(1, self._hold - self._warm)
        return MAX_BITS + frac * (self.budget - MAX_BITS)   # linear 8 -> B*

    # -- main hook (call once per step) ------------------------------------
    def step(self, global_step: int):
        for n in self.cooldown:
            if self.cooldown[n] > 0:
                self.cooldown[n] -= 1
        audit_every = ANNEAL_AUDIT_EVERY if global_step < self._hold else HOLD_AUDIT_EVERY
        if global_step < self._warm or global_step % audit_every != 0:
            return
        audit = self.audit_fn()
        base_kl = audit.pop("__base_kl__", {}).get("base_kl", 0.0)

        if self.recovery:
            self._do_recovery(audit, global_step)
        self._do_drops(audit, global_step)

        self._log_state(global_step, base_kl)

    # -- recovery (raise) --------------------------------------------------
    def _do_recovery(self, audit: dict, step: int):
        ups = [r["marginal_up"] for r in audit.values()
               if r.get("marginal_up") is not None]
        if not ups:
            return
        thresh = float(np.percentile(ups, RECOVER_PCTL))
        if thresh <= 0:
            return
        for n, r in sorted(audit.items(),
                           key=lambda kv: -(kv[1].get("marginal_up") or 0)):
            mu = r.get("marginal_up")
            if mu is None or mu < thresh:
                break
            m = self.layers[n]
            if self.cooldown[n] > 0 or self.raise_count[n] >= RAISE_CAP:
                continue
            if m.bit_width is not None and m.bit_width < MAX_BITS:
                newb = _next_higher(m.bit_width)
                self._change(n, m.bit_width, newb, step, "raise", mu)
                self.raise_count[n] += 1

    # -- drops to meet budget ----------------------------------------------
    def _do_drops(self, audit: dict, step: int):
        target = self.target_bits(step)
        guard = 0
        while avg_bits(self.model) > target and guard < len(self.layers):
            guard += 1
            best = None  # (cost_per_bit, name, newb)
            for n, r in audit.items():
                m = self.layers[n]
                # cooldown gates drops in the hold phase only; to keep pace with
                # the descending target, drops ignore cooldown during anneal
                if self.cooldown[n] > 0 and step >= self._hold:
                    continue
                b = m.bit_width
                if b is None or b <= MIN_BITS:
                    continue
                md = r.get("marginal_down")
                if md is None:
                    continue
                newb = _next_lower(b)
                bits_saved = b - newb
                cost = md / max(bits_saved, 1)
                if best is None or cost < best[0]:
                    best = (cost, n, newb)
            if best is None:
                break
            _, n, newb = best
            self._change(n, self.layers[n].bit_width, newb, step, "drop",
                         audit[n].get("marginal_down"))

    # -- helpers -----------------------------------------------------------
    def _change(self, name, frm, to, step, action, marginal):
        self.layers[name].set_bits(to)
        self.cooldown[name] = COOLDOWN_STEPS
        self.events.append(dict(step=step, layer=name, action=action,
                                from_bits=frm, to_bits=to,
                                marginal_kl=float(marginal) if marginal is not None else None))

    def _log_state(self, step, base_kl):
        self.events.append(dict(step=step, action="state",
                                avg_bits=avg_bits(self.model), base_kl=base_kl,
                                bits={n: m.bit_width for n, m in self.layers.items()}))

    def final_bitmap(self) -> Dict[str, int]:
        return {n: m.bit_width for n, m in self.layers.items()}
