# src/graph/static_converter.py

import torch
import torch.nn as nn
from torchao.quantization.qat.linear import Int8DynActInt4WeightQATLinear

# src/graph/static_converter.py

class StaticScaleLinear(nn.Module):
    def __init__(self, qat_linear, frozen_scale):
        super().__init__()
        self.in_features  = qat_linear.in_features
        self.out_features = qat_linear.out_features
        self.weight = nn.Parameter(qat_linear.weight.clone())
        self.bias   = nn.Parameter(qat_linear.bias.clone()) \
                      if qat_linear.bias is not None else None
        self.register_buffer('frozen_scale',
                             torch.tensor(frozen_scale, dtype=torch.float32))

    def forward(self, x):
        # frozen_scale is a scalar — works for any input shape (2D or 3D)
        s     = self.frozen_scale
        x_int8 = torch.clamp(torch.round(x / s), -128, 127)
        x_fake = x_int8 * s
        return nn.functional.linear(x_fake, self.weight, self.bias)

def convert_stable_layers(model, labels, include_borderline=False):
    """
    Replaces STATIC_OK layers (and optionally BORDERLINE) 
    with StaticScaleLinear modules.
    
    Returns modified model and conversion report.
    """
    report = {'converted': [], 'kept_dynamic': [], 'errors': []}

    target_verdicts = {'STATIC_OK'}
    if include_borderline:
        target_verdicts.add('BORDERLINE')

    for name, info in labels.items():
        if info['verdict'] not in target_verdicts:
            report['kept_dynamic'].append(name)
            continue

        # navigate to parent module
        parts  = name.split('.')
        parent = model
        for part in parts[:-1]:
            parent = getattr(parent, part)
        attr   = parts[-1]
        module = getattr(parent, attr)

        if not isinstance(module, Int8DynActInt4WeightQATLinear):
            report['errors'].append(f"{name}: not a QAT linear")
            continue

        try:
            static_mod = StaticScaleLinear(module, frozen_scale=info['mean'])
            setattr(parent, attr, static_mod)
            report['converted'].append({
                'name': name, 'cv': info['cv'],
                'frozen_scale': info['mean']
            })
        except Exception as e:
            report['errors'].append(f"{name}: {e}")
            report['kept_dynamic'].append(name)

    return model, report


def print_conversion_report(report):
    print(f"\nCONVERSION REPORT")
    print(f"  Converted to static: {len(report['converted'])}")
    print(f"  Kept dynamic:        {len(report['kept_dynamic'])}")
    if report['errors']:
        print(f"  Errors:              {len(report['errors'])}")
        for e in report['errors']:
            print(f"    {e}")