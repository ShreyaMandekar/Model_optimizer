"""4-state controller implementing all 4 QAT freeze/unfreeze strategies.

States (per layer):
  DYNAMIC → TENTATIVE_FROZEN → CONFIRMED_FROZEN
                │                    │
                └── COOLDOWN ←───────┘
                      │
                      └→ DYNAMIC (unless re-freeze cap hit → pinned DYNAMIC)

Strategies (TASK.md §5):
  s1_oneway   — freeze on trigger, never unfreeze
  s2_recal    — s1 + single end-of-training recalibrate
  s3_reactive — full controller (freeze + unfreeze during training + recalibrate)
  s4_soft     — high-momentum EMA scale during training, hard-freeze all at end

Metrics:
  cv   — freeze when CV < threshold (patience), unfreeze on drift
  kl   — freeze when KL_s->t < kl_freeze, unfreeze when KL_s->t > kl_unfreeze

Anti-thrashing (TASK.md §4.6):
  1. Dead band (asymmetric freeze/unfreeze thresholds)
  2. Cooldown M steps after unfreeze
  3. Re-freeze cap: max 2 unfreezes per layer; beyond → pinned DYNAMIC
"""

from __future__ import annotations
import enum
from typing import Dict, List, Optional, Set
import torch

from config import (
    CV_FREEZE_THRESHOLD, CV_FREEZE_PATIENCE,
    KL_FREEZE_THRESHOLD, KL_UNFREEZE_THRESHOLD,
    CV_DRIFT_THRESHOLD, CV_DRIFT_PATIENCE,
    COOLDOWN_STEPS, REFREEZE_CAP, AUDIT_EVERY,
    FREEZE_PHASE_GATE, REGRESSION_TOL, SOFT_EMA_MOMENTUM,
    TARGET_FROZEN_FRACTION,
)
from qat_linear import DynStatQATLinear


class LayerState(enum.Enum):
    DYNAMIC           = "DYNAMIC"
    TENTATIVE_FROZEN  = "TENTATIVE_FROZEN"
    CONFIRMED_FROZEN  = "CONFIRMED_FROZEN"
    COOLDOWN          = "COOLDOWN"
    PINNED_DYNAMIC    = "PINNED_DYNAMIC"   # re-freeze cap exhausted


class Controller:
    def __init__(
        self,
        model: torch.nn.Module,
        metric: str,          # "cv" or "kl"
        strategy: str,        # "s1_oneway" | "s2_recal" | "s3_reactive" | "s4_soft"
        total_steps: int,
        tracker=None,         # OnlineStabilityTracker (needed for CV drift)
        kl_fn=None,           # callable(layer_names) -> {name: kl_score}
        val_fn=None,          # callable() -> float (validation loss for regression check)
    ):
        assert metric   in ("cv", "kl"),          f"Unknown metric: {metric}"
        assert strategy in ("s1_oneway", "s2_recal", "s3_reactive", "s4_soft"), \
            f"Unknown strategy: {strategy}"

        self.model         = model
        self.metric        = metric
        self.strategy      = strategy
        self.total_steps   = total_steps
        self.tracker       = tracker
        self.kl_fn         = kl_fn
        self.val_fn        = val_fn

        # collect all DynStatQATLinear layers
        self._layers: Dict[str, DynStatQATLinear] = {
            name: m for name, m in model.named_modules()
            if isinstance(m, DynStatQATLinear)
        }
        self.n_layers = len(self._layers)

        # per-layer controller state
        self.state:            Dict[str, LayerState] = {n: LayerState.DYNAMIC for n in self._layers}
        self.cooldown_left:    Dict[str, int]         = {n: 0 for n in self._layers}
        self.unfreeze_count:   Dict[str, int]         = {n: 0 for n in self._layers}
        self.cv_stable_steps:  Dict[str, int]         = {n: 0 for n in self._layers}
        self.drift_audit_hits: Dict[str, int]         = {n: 0 for n in self._layers}
        self.tentative_audit_count: Dict[str, int]    = {n: 0 for n in self._layers}

        # frozen scale snapshot per layer (for drift computation)
        self.frozen_scale_snap: Dict[str, float]      = {}

        # soft-freeze: per-layer soft EMA scale
        self.soft_scale: Dict[str, Optional[float]]   = {n: None for n in self._layers}

        # target count
        self.target_frozen = max(1, int(TARGET_FROZEN_FRACTION * self.n_layers))

        # last val loss (for regression check)
        self._last_val_loss: Optional[float] = None

        self._step = 0
        self._freeze_phase_start = int(FREEZE_PHASE_GATE * total_steps)

        # event log
        self.events: List[dict] = []

    # ------------------------------------------------------------------
    # Main entry: call once per training step
    # ------------------------------------------------------------------

    def step(self, tracker_states: Optional[dict] = None):
        self._step += 1

        if self.strategy == "s4_soft":
            self._update_soft_scales(tracker_states)
            return

        if self._step < self._freeze_phase_start:
            return

        # ---- per-layer freeze eligibility check ----
        if self.metric == "cv":
            self._cv_freeze_check(tracker_states or {})
        # KL freeze check happens in audit (not every step — too expensive)

        # ---- periodic audit ----
        if self._step % AUDIT_EVERY == 0:
            self._run_audit()

        # ---- cooldown countdown ----
        for name in self._layers:
            if self.state[name] == LayerState.COOLDOWN:
                self.cooldown_left[name] -= 1
                if self.cooldown_left[name] <= 0:
                    self.state[name] = LayerState.DYNAMIC
                    self._log("cooldown_end", name)

    # ------------------------------------------------------------------
    # CV freeze check (called every step)
    # ------------------------------------------------------------------

    def _cv_freeze_check(self, tracker_states: dict):
        if self.metric != "cv":
            return
        for name, layer in self._layers.items():
            if self.state[name] not in (LayerState.DYNAMIC,):
                continue
            st = tracker_states.get(name, {})
            cv = st.get("cv", 999.0)
            if cv < CV_FREEZE_THRESHOLD:
                self.cv_stable_steps[name] += 1
            else:
                self.cv_stable_steps[name] = 0
            if self.cv_stable_steps[name] >= CV_FREEZE_PATIENCE:
                self._do_freeze(name, reason="cv_trigger")
                self.cv_stable_steps[name] = 0

    # ------------------------------------------------------------------
    # Periodic audit (both metrics)
    # ------------------------------------------------------------------

    def _run_audit(self):
        # 1. TENTATIVE → CONFIRMED after surviving 1 audit
        for name in list(self._layers):
            if self.state[name] == LayerState.TENTATIVE_FROZEN:
                self.tentative_audit_count[name] += 1
                if self.tentative_audit_count[name] >= 1:
                    self.state[name] = LayerState.CONFIRMED_FROZEN
                    self._log("confirmed_frozen", name)

        # 2. KL freeze check for dynamic/cooldown layers
        if self.metric == "kl" and self.kl_fn is not None:
            eligible = [n for n, s in self.state.items()
                        if s == LayerState.DYNAMIC and self._step >= self._freeze_phase_start]
            if eligible:
                kl_scores = self.kl_fn(eligible)
                for name, kl in kl_scores.items():
                    if kl < KL_FREEZE_THRESHOLD:
                        self._do_freeze(name, reason=f"kl_trigger kl={kl:.4f}")

        # 3. Unfreeze audit (s3_reactive only)
        if self.strategy == "s3_reactive":
            self._unfreeze_audit()

    def _unfreeze_audit(self):
        frozen_names = [n for n, s in self.state.items()
                        if s in (LayerState.TENTATIVE_FROZEN, LayerState.CONFIRMED_FROZEN)]
        if not frozen_names:
            return

        if self.metric == "cv":
            # drift-based unfreeze
            for name in frozen_names:
                snap = self.frozen_scale_snap.get(name, 0.0)
                if snap > 0 and self.tracker is not None:
                    drift = self.tracker.drift(name, snap)
                    if drift > CV_DRIFT_THRESHOLD:
                        self.drift_audit_hits[name] += 1
                    else:
                        self.drift_audit_hits[name] = 0
                    if self.drift_audit_hits[name] >= CV_DRIFT_PATIENCE:
                        self._do_unfreeze(name, reason=f"cv_drift drift={drift:.3f}")
                        self.drift_audit_hits[name] = 0

        elif self.metric == "kl" and self.kl_fn is not None:
            kl_scores = self.kl_fn(frozen_names)
            for name, kl in kl_scores.items():
                if kl > KL_UNFREEZE_THRESHOLD:
                    self._do_unfreeze(name, reason=f"kl_unfreeze kl={kl:.4f}")

    # ------------------------------------------------------------------
    # Soft-freeze (s4): update per-layer slow EMA
    # ------------------------------------------------------------------

    def _update_soft_scales(self, tracker_states: Optional[dict]):
        if tracker_states is None:
            return
        for name, layer in self._layers.items():
            st = tracker_states.get(name, {})
            raw = st.get("raw_scale")
            if raw is None:
                continue
            if self.soft_scale[name] is None:
                self.soft_scale[name] = raw
            else:
                m = SOFT_EMA_MOMENTUM
                self.soft_scale[name] = m * self.soft_scale[name] + (1 - m) * raw
            # update the layer's activation_fake_quantizer scale to soft EMA
            # (not fully frozen — scale drifts slowly)
            layer._last_mean_scale = self.soft_scale[name]

    # ------------------------------------------------------------------
    # Freeze / unfreeze primitives
    # ------------------------------------------------------------------

    def _do_freeze(self, name: str, reason: str = ""):
        if self.state[name] not in (LayerState.DYNAMIC,):
            return
        layer = self._layers[name]
        scale = layer._last_mean_scale
        if scale is None:
            return
        layer.freeze(scale=scale)
        self.frozen_scale_snap[name] = scale
        self.state[name] = LayerState.TENTATIVE_FROZEN
        self.tentative_audit_count[name] = 0
        self._log("freeze", name, extra={"reason": reason, "scale": scale})

    def _do_unfreeze(self, name: str, reason: str = ""):
        if self.state[name] not in (LayerState.TENTATIVE_FROZEN, LayerState.CONFIRMED_FROZEN):
            return
        if self.unfreeze_count[name] >= REFREEZE_CAP:
            # pin to DYNAMIC — stop auditing this layer
            self._layers[name].unfreeze()
            self.state[name] = LayerState.PINNED_DYNAMIC
            self._log("pinned_dynamic", name, extra={"reason": "refreeze_cap_hit"})
            return
        self._layers[name].unfreeze()
        self.state[name] = LayerState.COOLDOWN
        self.cooldown_left[name] = COOLDOWN_STEPS
        self.unfreeze_count[name] += 1
        self._log("unfreeze", name, extra={"reason": reason, "count": self.unfreeze_count[name]})

    # ------------------------------------------------------------------
    # Regression safety net
    # ------------------------------------------------------------------

    def check_regression(self, current_val_loss: float):
        if self._last_val_loss is None:
            self._last_val_loss = current_val_loss
            return
        regression = (current_val_loss - self._last_val_loss) / (self._last_val_loss + 1e-8)
        if regression > REGRESSION_TOL and self.strategy in ("s3_reactive", "s2_recal"):
            self._log("regression_trigger", "global",
                      extra={"prev": self._last_val_loss, "curr": current_val_loss})
            # immediate full audit of frozen layers
            if self.strategy == "s3_reactive":
                self._unfreeze_audit()
        self._last_val_loss = current_val_loss

    # ------------------------------------------------------------------
    # End-of-training recalibrate (s2 and s3)
    # ------------------------------------------------------------------

    def final_recalibrate(self, calib_forward_fn):
        """Unfreeze all → run calib forward → re-freeze with fresh scales."""
        if self.strategy not in ("s2_recal", "s3_reactive", "s4_soft"):
            return
        # unfreeze all
        for name, layer in self._layers.items():
            if layer.frozen:
                layer.unfreeze()
        # one forward pass to refresh scales
        calib_forward_fn()
        # re-freeze to target fraction by KL (lowest KL = safest to freeze)
        if self.kl_fn is not None:
            kl_scores = self.kl_fn(list(self._layers.keys()))
            ranked = sorted(kl_scores, key=lambda n: kl_scores[n])
        else:
            # fall back to CV ordering
            ranked = list(self._layers.keys())

        for i, name in enumerate(ranked):
            if i < self.target_frozen:
                layer = self._layers[name]
                scale = layer._last_mean_scale or 1.0
                layer.freeze(scale=scale)
                self.state[name] = LayerState.CONFIRMED_FROZEN
                self._log("recalibrate_freeze", name)
            else:
                self.state[name] = LayerState.DYNAMIC

    # ------------------------------------------------------------------
    # Hard-freeze all for s4_soft at end of training
    # ------------------------------------------------------------------

    def finalize_soft(self):
        if self.strategy != "s4_soft":
            return
        for name, layer in self._layers.items():
            s = self.soft_scale[name] or layer._last_mean_scale or 1.0
            layer.freeze(scale=s)
            self.state[name] = LayerState.CONFIRMED_FROZEN
            self._log("soft_hardfreeze", name, extra={"scale": s})

    # ------------------------------------------------------------------
    # State queries
    # ------------------------------------------------------------------

    def frozen_names(self) -> List[str]:
        return [n for n, s in self.state.items()
                if s in (LayerState.TENTATIVE_FROZEN, LayerState.CONFIRMED_FROZEN)]

    def frozen_fraction(self) -> float:
        return len(self.frozen_names()) / max(1, self.n_layers)

    def state_summary(self) -> dict:
        counts = {}
        for s in LayerState:
            counts[s.value] = sum(1 for v in self.state.values() if v == s)
        return counts

    # ------------------------------------------------------------------
    # Logging
    # ------------------------------------------------------------------

    def _log(self, event_type: str, name: str, extra: Optional[dict] = None):
        entry = {"step": self._step, "event": event_type, "layer": name}
        if extra:
            entry.update(extra)
        self.events.append(entry)
