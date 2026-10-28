"""CPU-only numerical gate for Objective 2 feature, graph and logit invariance."""

import argparse
import json
from pathlib import Path

import torch

from models.summer_dynamic import SummerDynamicWHAR
from o2_features import WIDTHS, features, random_rotations, rotate_raw
from workstation_experiments import write_json


def check():
    torch.set_num_threads(2)
    torch.manual_seed(2026)
    raw = torch.randn(2, 50, 9, 6)
    available = torch.ones(2, 9, dtype=torch.bool)
    rotations = random_rotations(2, 9, raw.device)
    rotated = rotate_raw(raw, rotations)
    report = {'device': 'cpu', 'rotation': 'independent fixed SO(3) per window/IMU',
              'feature_atol_rtol': 1e-5, 'model_atol_rtol': 1e-4, 'cases': []}
    for kind in ('magnitude', 'mag_dot'):
        original = features(raw, kind)
        changed = features(rotated, kind)
        mean = original.mean((0, 1), keepdim=True)
        std = original.std((0, 1), keepdim=True).clamp_min(1e-6)
        x = (original - mean) / std
        rx = (changed - mean) / std
        feature_ok = torch.allclose(x, rx, atol=1e-5, rtol=1e-5)
        for graph in ('physical', 'residual', 'none'):
            torch.manual_seed(13)
            model = SummerDynamicWHAR(in_features=WIDTHS[kind],
                                      graph_variant=graph).eval()
            with torch.inference_mode():
                logits, _, diagnostics = model(x, available,
                                                return_diagnostics=True)
                rotated_logits, _, rotated_diagnostics = model(
                    rx, available, return_diagnostics=True)
            adjacency = diagnostics['adjacency'].float()
            rotated_adjacency = rotated_diagnostics['adjacency'].float()
            adjacency_ok = torch.allclose(adjacency, rotated_adjacency,
                                          atol=1e-4, rtol=1e-4)
            logits_ok = torch.allclose(logits.float(), rotated_logits.float(),
                                       atol=1e-4, rtol=1e-4)
            report['cases'].append({'feature': kind, 'graph': graph,
                'feature_max_abs': float((x - rx).abs().max()),
                'adjacency_max_abs': float((adjacency - rotated_adjacency).abs().max()),
                'logit_max_abs': float((logits - rotated_logits).abs().max()),
                'feature_passed': bool(feature_ok),
                'adjacency_passed': bool(adjacency_ok),
                'logits_passed': bool(logits_ok)})
    report['passed'] = all(all(case[key] for key in
        ('feature_passed', 'adjacency_passed', 'logits_passed'))
        for case in report['cases'])
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    report = check()
    write_json(Path(args.output), report)
    print(json.dumps(report, indent=2))
    raise SystemExit(0 if report['passed'] else 2)
