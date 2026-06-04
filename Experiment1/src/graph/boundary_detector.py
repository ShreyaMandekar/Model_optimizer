# src/graph/boundary_detector.py

import statistics
import torch
from torchao.quantization.qat.linear import Int8DynActInt4WeightQATLinear

class ActivationStabilityDetector:
    """
    Measures how much each layer's dynamic quantization scale
    varies across real inputs. Stable layers (low CV) are candidates
    for static quantization — eliminating choose_qparams overhead.
    
    This replaces the fusion-boundary approach: the real overhead
    is not graph fragmentation but redundant dynamic scale computation.
    """

    def __init__(self, cv_threshold_static=5.0, cv_threshold_dynamic=15.0):
        self.cv_threshold_static  = cv_threshold_static   # below = STATIC OK
        self.cv_threshold_dynamic = cv_threshold_dynamic  # above = must stay DYNAMIC
        self.scale_history = {}
        self.hooks = []

    def attach_hooks(self, model):
        """Attach forward hooks to all QAT linear layers."""
        self.scale_history = {}
        self.hooks = []

        for name, module in model.named_modules():
            if isinstance(module, Int8DynActInt4WeightQATLinear):
                self.scale_history[name] = []
                self.hooks.append(
                    module.register_forward_hook(self._make_hook(name))
                )
        print(f"Attached hooks to {len(self.hooks)} QAT layers")
        return self

    def _make_hook(self, name):
        def hook(module, input, output):
            x = input[0].detach().float()
            scale = x.abs().amax(dim=-1) / 127.0
            self.scale_history[name].append(scale.mean().item())
        return hook

    def remove_hooks(self):
        for h in self.hooks:
            h.remove()
        self.hooks = []

    def compute_labels(self):
        """
        Returns dict: layer_name -> {
            'mean': float, 'std': float, 'cv': float,
            'verdict': 'STATIC_OK' | 'BORDERLINE' | 'DYNAMIC',
            'n_samples': int
        }
        """
        labels = {}
        for name, scales in self.scale_history.items():
            if len(scales) < 2:
                continue
            mean = statistics.mean(scales)
            std  = statistics.stdev(scales)
            cv   = (std / mean * 100) if mean > 0 else 0

            if cv < self.cv_threshold_static:
                verdict = 'STATIC_OK'
            elif cv < self.cv_threshold_dynamic:
                verdict = 'BORDERLINE'
            else:
                verdict = 'DYNAMIC'

            labels[name] = {
                'mean': mean, 'std': std, 'cv': cv,
                'verdict': verdict, 'n_samples': len(scales)
            }
        return labels

    def summary(self, labels):
        static   = sum(1 for v in labels.values() if v['verdict'] == 'STATIC_OK')
        border   = sum(1 for v in labels.values() if v['verdict'] == 'BORDERLINE')
        dynamic  = sum(1 for v in labels.values() if v['verdict'] == 'DYNAMIC')
        total    = len(labels)

        # cost estimate from your profiler numbers
        ms_choose  = 27.0 / 120   # per-call cost of choose_qparams
        ms_amin    = 14.0 / 120   # per-call cost of aten::amin
        ms_saved   = static * (ms_choose + ms_amin)

        print(f"\n{'='*60}")
        print(f"  ACTIVATION STABILITY SUMMARY")
        print(f"{'='*60}")
        print(f"  STATIC OK  (cv <  5%): {static:3d}/{total}")
        print(f"  BORDERLINE (cv 5-15%): {border:3d}/{total}")
        print(f"  DYNAMIC    (cv > 15%): {dynamic:3d}/{total}")
        print(f"\n  Estimated savings if STATIC_OK layers frozen:")
        print(f"  choose_qparams + amin eliminated: {static} calls")
        print(f"  Time saved per inference:         {ms_saved:.1f}ms")
        print(f"  Current QAT compiled latency:     69.6ms")
        print(f"  Projected latency:                {69.6 - ms_saved:.1f}ms")
        print(f"{'='*60}")

        return {'static': static, 'borderline': border,
                'dynamic': dynamic, 'ms_saved': ms_saved}