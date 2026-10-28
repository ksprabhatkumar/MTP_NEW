"""Real-fold split, rotation and preprocessing gate for the next summer O2 run."""
import argparse
import json
from pathlib import Path

import torch

from models.summer_dynamic import SummerDynamicWHAR
from o2_features import O2Corpus, features, random_rotations, rotate_raw, WIDTHS
from workstation_experiments import write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    torch.set_num_threads(8)
    torch.manual_seed(20261021)
    corpus = O2Corpus()
    report = {'device': str(torch.device('cuda')), 'folds': [], 'rotation': {}, 'invariance': [],
              'excluded': sorted(corpus.valid_manifest['excluded']),
              'retained_windows': corpus.valid_manifest['retained_windows']}
    for subject in (2, 5, 15):
        fold = corpus.fold(subject, 'raw')
        summary = fold['summary']
        if subject in summary['train_subjects'] or summary['validation_subject'] in summary['train_subjects']:
            raise RuntimeError('Subject leakage in training fold')
        if summary['normalization'] != 'feature transform before train-only mean/std by sensor/feature':
            raise RuntimeError('Unexpected normalization policy')
        report['folds'].append({key: summary[key] for key in
            ('test_subject', 'validation_subject', 'train_subjects', 'counts',
             'common_validation_classes', 'excluded_recordings', 'normalization')})
        if subject != 2:
            del fold
            torch.cuda.empty_cache()
            continue
        part = fold['data']['val_ideal']
        ix = torch.linspace(0, len(part['y'])-1, min(64, len(part['y'])), device='cuda').long()
        raw = part['raw'][ix]
        available = part['a'][ix]
        rotations = random_rotations(len(raw), 9, raw.device)
        identity = torch.eye(3, device=raw.device)
        orthogonality = torch.matmul(rotations.transpose(-1, -2), rotations)
        determinant = torch.linalg.det(rotations)
        angle = torch.acos(((rotations.diagonal(dim1=-2, dim2=-1).sum(-1)-1)/2).clamp(-1, 1))
        rotated = rotate_raw(raw, rotations)
        report['rotation'] = {
            'orthogonality_max_abs': float((orthogonality-identity).abs().max()),
            'determinant_min': float(determinant.min()),
            'determinant_max': float(determinant.max()),
            'angle_deg_min': float(torch.rad2deg(angle).min()),
            'angle_deg_median': float(torch.rad2deg(angle).median()),
            'angle_deg_max': float(torch.rad2deg(angle).max()),
            'acceleration_rotation_max_abs': float((rotated[..., :3] - torch.einsum(
                'bnij,btnj->btni', rotations, raw[..., :3])).abs().max()),
            'gyro_rotation_max_abs': float((rotated[..., 3:] - torch.einsum(
                'bnij,btnj->btni', rotations, raw[..., 3:])).abs().max()),
        }
        for kind in ('magnitude', 'mag_dot'):
            clean = features(raw, kind)
            changed = features(rotated, kind)
            train_raw = fold['data']['train']['raw']
            train_features = features(train_raw, kind)
            mean = train_features.mean((0, 1), keepdim=True)
            std = train_features.std((0, 1), keepdim=True).clamp_min(1e-6)
            x = ((clean-mean)/std)*available[:, None, :, None]
            rx = ((changed-mean)/std)*available[:, None, :, None]
            feature_error = float((x-rx).abs().max())
            first = clean[:, 0]
            first_lag_error = 0.0
            if kind == 'mag_dot':
                first_lag_error = float(torch.maximum(
                    (first[..., 3] - (raw[:, 0, :, :3]**2).sum(-1)).abs().max(),
                    (first[..., 4] - (raw[:, 0, :, 3:]**2).sum(-1)).abs().max()))
            for graph in ('none', 'physical', 'residual'):
                torch.manual_seed(13)
                model = SummerDynamicWHAR(in_features=WIDTHS[kind], graph_variant=graph).cuda().eval()
                with torch.inference_mode():
                    logits, _, diagnostics = model(x, available, return_diagnostics=True)
                    other, _, other_diagnostics = model(rx, available, return_diagnostics=True)
                report['invariance'].append({'feature': kind, 'graph': graph,
                    'feature_max_abs': feature_error, 'first_lag_max_abs': first_lag_error,
                    'adjacency_max_abs': float((diagnostics['adjacency']-other_diagnostics['adjacency']).abs().max()),
                    'logit_max_abs': float((logits-other).abs().max())})
                del model
        del fold
        torch.cuda.empty_cache()
    report['passed'] = (report['retained_windows'] == 76268
        and report['rotation']['orthogonality_max_abs'] < 1e-5
        and report['rotation']['determinant_min'] > 0.99999
        and all(row['feature_max_abs'] < 1e-3 and row['adjacency_max_abs'] < 1e-4
                and row['logit_max_abs'] < 1e-3 and row['first_lag_max_abs'] < 1e-5
                for row in report['invariance']))
    write_json(Path(args.output), report)
    print(json.dumps({'output': args.output, 'passed': report['passed'],
                      'max_feature_error': max(row['feature_max_abs'] for row in report['invariance']),
                      'max_logit_error': max(row['logit_max_abs'] for row in report['invariance'])}))
    raise SystemExit(0 if report['passed'] else 2)


if __name__ == '__main__':
    main()
