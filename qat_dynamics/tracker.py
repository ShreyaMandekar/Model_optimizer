"""Per-layer activation scale tracker (logging-only + drift signal).

Forked from Experiment1/src/training/online_stability_tracker.py.
Changes vs original:
  - Logging-only: does NOT trigger freezes itself; the controller does.
  - Tracks a running "shadow amax" (what the dynamic scale WOULD be) even for
    frozen layers, so the drift-based CV-column unfreeze signal is available.
  - Stores full per-step log for parquet export.
  - Does NOT import from Experiment1.
"""

import collections
from typing import Dict, List, Optional
import torch
from qat_linear import DynStatQATLinear


class OnlineStabilityTracker:
    def __init__(self, ema_alpha: float = 0.15):
        self.ema_alpha = ema_alpha

        # per-layer EMA state
        self.ema_mean:  Dict[str, Optional[float]] = {}
        self.ema_var:   Dict[str, Optional[float]] = {}
        # shadow scale: what the dynamic scale would be (computed even when frozen)
        self.shadow_scale: Dict[str, Optional[float]] = {}
        self.shadow_ema:   Dict[str, Optional[float]] = {}

        self._current_scales: Dict[str, float] = {}
        self.hooks: List = []

        # full trajectory log: list of dicts
        self.log: List[dict] = []
        self._step = 0

    # ------------------------------------------------------------------
    # Hook attachment
    # ------------------------------------------------------------------

    def attach_hooks(self, model: torch.nn.Module) -> "OnlineStabilityTracker":
        self.remove_hooks()
        for name, module in model.named_modules():
            if isinstance(module, DynStatQATLinear):
                self.ema_mean[name]    = None
                self.ema_var[name]     = None
                self.shadow_scale[name] = None
                self.shadow_ema[name]  = None
                self.hooks.append(
                    module.register_forward_hook(self._make_hook(name, module))
                )
        return self

    def _make_hook(self, name: str, module: DynStatQATLinear):
        def hook(mod, inp, out):
            x = inp[0].detach().float()
            # always compute the "would-be" dynamic scale
            raw_scale = (x.abs().amax(dim=-1) / 127.0).mean().item()
            self._current_scales[name] = raw_scale
        return hook

    # ------------------------------------------------------------------
    # Step update — call once per training batch after forward
    # ------------------------------------------------------------------

    def step(self, frozen_layers: Optional[set] = None) -> Dict[str, dict]:
        """Update EMAs; return per-layer state dict."""
        frozen_layers = frozen_layers or set()
        self._step += 1
        states = {}

        for name, raw in self._current_scales.items():
            # update shadow EMA (always tracks what dynamic scale would be)
            if self.shadow_ema[name] is None:
                self.shadow_ema[name] = raw
            else:
                a = self.ema_alpha
                self.shadow_ema[name] = a * raw + (1 - a) * self.shadow_ema[name]
            self.shadow_scale[name] = raw

            # update main EMA (same as shadow when not frozen)
            is_frozen = name in frozen_layers
            if not is_frozen:
                if self.ema_mean[name] is None:
                    self.ema_mean[name] = raw
                    self.ema_var[name]  = 0.0
                else:
                    a = self.ema_alpha
                    prev = self.ema_mean[name]
                    self.ema_mean[name] = a * raw + (1 - a) * prev
                    self.ema_var[name]  = (1 - a) * (
                        self.ema_var[name] + a * (raw - prev) ** 2
                    )

            mean = self.ema_mean[name] or raw
            std  = (self.ema_var[name] or 0.0) ** 0.5
            cv   = (std / mean * 100) if mean > 1e-8 else 0.0

            states[name] = {
                "step":         self._step,
                "raw_scale":    raw,
                "ema_mean":     self.ema_mean[name],
                "ema_var":      self.ema_var[name],
                "cv":           cv,
                "shadow_ema":   self.shadow_ema[name],
                "is_frozen":    is_frozen,
            }

        self._current_scales = {}
        return states

    # ------------------------------------------------------------------
    # Drift signal: |shadow_ema − frozen_scale| / frozen_scale
    # ------------------------------------------------------------------

    def drift(self, name: str, frozen_scale: float) -> float:
        if self.shadow_ema[name] is None or frozen_scale < 1e-8:
            return 0.0
        return abs(self.shadow_ema[name] - frozen_scale) / frozen_scale

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    def get_cv(self, name: str) -> float:
        mean = self.ema_mean.get(name)
        var  = self.ema_var.get(name)
        if mean is None or mean < 1e-8:
            return 0.0
        return ((var or 0.0) ** 0.5 / mean * 100)

    def remove_hooks(self):
        for h in self.hooks:
            h.remove()
        self.hooks = []

    def layer_names(self) -> List[str]:
        return list(self.ema_mean.keys())
