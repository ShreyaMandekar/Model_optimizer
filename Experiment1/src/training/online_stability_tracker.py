import torch
import collections
from torchao.quantization.qat.linear import Int8DynActInt4WeightQATLinear


class OnlineStabilityTracker:
    """
    Tracks per-layer activation scale variance during training.
    Uses exponential moving average so recent batches matter more.
    Declares a layer freeze-eligible when EMA CV stays below
    threshold for `patience` consecutive steps.
    """

    def __init__(self,
             cv_threshold=5.0,
             patience=20,
             ema_alpha=0.15):      # was 0.05 — faster response
        self.cv_threshold = cv_threshold
        self.patience     = patience
        self.ema_alpha    = ema_alpha  # lower = slower to update

        # per-layer state
        self.ema_mean     = {}   # exponential moving average of scale
        self.ema_var      = {}   # exponential moving average of variance
        self.stable_steps = {}   # consecutive steps below threshold
        self.frozen       = {}   # layers declared frozen
        self.hooks        = []
        self._current_scales = {}

    def attach_hooks(self, model):
        self.hooks = []
        for name, module in model.named_modules():
            if isinstance(module, Int8DynActInt4WeightQATLinear):
                self.ema_mean[name]     = None
                self.ema_var[name]      = None
                self.stable_steps[name] = 0
                self.frozen[name]       = False
                self.hooks.append(
                    module.register_forward_hook(self._make_hook(name))
                )
        return self

    def _make_hook(self, name):
        def hook(module, input, output):
            if self.frozen[name]:
                return  # already frozen, stop tracking
            x = input[0].detach().float()
            scale = (x.abs().amax(dim=-1) / 127.0).mean().item()
            self._current_scales[name] = scale
        return hook

    def step(self):
        """
        Call once per training batch after forward pass.
        Updates EMA estimates and checks freeze eligibility.
        Returns list of newly freeze-eligible layer names.
        """
        newly_eligible = []

        for name, scale in self._current_scales.items():
            if self.frozen[name]:
                continue

            # initialise EMA on first observation
            if self.ema_mean[name] is None:
                self.ema_mean[name] = scale
                self.ema_var[name]  = 0.0
                continue

            # update EMA mean and variance
            alpha = self.ema_alpha
            prev_mean = self.ema_mean[name]
            self.ema_mean[name] = alpha * scale + (1 - alpha) * prev_mean
            self.ema_var[name]  = (1 - alpha) * (
                self.ema_var[name] + alpha * (scale - prev_mean) ** 2
            )

            # compute current CV estimate
            mean = self.ema_mean[name]
            std  = self.ema_var[name] ** 0.5
            cv   = (std / mean * 100) if mean > 1e-8 else 0.0

            if cv < self.cv_threshold:
                self.stable_steps[name] += 1
            else:
                self.stable_steps[name] = 0  # reset if unstable

            if self.stable_steps[name] >= self.patience:
                newly_eligible.append(name)

        self._current_scales = {}
        return newly_eligible

    def declare_frozen(self, name):
        self.frozen[name] = True
        self.stable_steps[name] = 0

    def get_freeze_candidates(self):
        """Returns layers eligible to freeze, sorted by stability."""
        return sorted(
            [(n, self.stable_steps[n]) for n, f in self.frozen.items()
             if not f and self.stable_steps[n] >= self.patience],
            key=lambda x: -x[1]
        )

    def remove_hooks(self):
        for h in self.hooks:
            h.remove()
        self.hooks = []

    def summary(self):
        frozen  = sum(self.frozen.values())
        dynamic = len(self.frozen) - frozen
        print(f"Frozen: {frozen}/{len(self.frozen)}  Dynamic: {dynamic}")
        for name, info in sorted(self.frozen.items()):
            cv = 0.0
            if self.ema_mean[name] and self.ema_var[name] is not None:
                cv = (self.ema_var[name]**0.5 / self.ema_mean[name] * 100
                      if self.ema_mean[name] > 1e-8 else 0.0)
            status = "FROZEN" if self.frozen[name] else f"cv={cv:.1f}%"
            print(f"  {name:<55} {status}")

