import torch
import copy
from Experiment1.src.graph.static_converter import StaticScaleLinear
from torchao.quantization.qat.linear import Int8DynActInt4WeightQATLinear


class ProgressiveFreezer:
    """
    Progressively converts QAT layers to static scale during training.
    Freezes in tiers: most stable first, with adaptation gaps between.
    """

    def __init__(self,
                 model,
                 tracker,
                 freeze_tier_size=5,
                 adaptation_steps=100):
        self.model            = model
        self.tracker          = tracker
        self.freeze_tier_size = freeze_tier_size   # freeze N layers at a time
        self.adaptation_steps = adaptation_steps   # steps between freeze tiers
        self.steps_since_last_freeze = 0
        self.total_frozen     = 0
        self.freeze_log       = []  # list of (step, layer_name, scale)

    def maybe_freeze(self, global_step):
        """
        Call once per training step.
        Returns True if any layers were frozen this step.
        """
        self.steps_since_last_freeze += 1

        # enforce adaptation gap between tiers
        if self.steps_since_last_freeze < self.adaptation_steps:
            return False

        candidates = self.tracker.get_freeze_candidates()
        if not candidates:
            return False

        # take top freeze_tier_size candidates
        to_freeze = candidates[:self.freeze_tier_size]
        frozen_this_step = []

        for name, stable_steps in to_freeze:
            success = self._freeze_layer(name)
            if success:
                self.tracker.declare_frozen(name)
                frozen_this_step.append(name)
                self.freeze_log.append({
                    'step': global_step,
                    'layer': name,
                    'scale': self.tracker.ema_mean[name]
                })

        if frozen_this_step:
            self.total_frozen += len(frozen_this_step)
            self.steps_since_last_freeze = 0
            print(f"\n[Step {global_step}] Froze {len(frozen_this_step)} layers "
                  f"(total frozen: {self.total_frozen}/38)")
            for n in frozen_this_step:
                scale = self.tracker.ema_mean[n]
                print(f"  {n:<55} scale={scale:.5f}")

        return len(frozen_this_step) > 0

    def _freeze_layer(self, name):
        """Replace QAT linear with StaticScaleLinear in-place."""
        parts  = name.split('.')
        parent = self.model
        for part in parts[:-1]:
            try:
                parent = getattr(parent, part)
            except AttributeError:
                return False

        attr   = parts[-1]
        module = getattr(parent, attr)

        if not isinstance(module, Int8DynActInt4WeightQATLinear):
            return False

        frozen_scale = self.tracker.ema_mean[name]
        static_mod   = StaticScaleLinear(module, frozen_scale)
        setattr(parent, attr, static_mod)
        return True

    def get_freeze_schedule(self):
        """Returns the log of when each layer was frozen."""
        return self.freeze_log