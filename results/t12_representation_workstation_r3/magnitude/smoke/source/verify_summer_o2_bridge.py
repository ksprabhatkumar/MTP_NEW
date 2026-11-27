"""Compare an O2 raw/physical bridge job with its frozen summer O1 twin."""

import argparse
import json
from pathlib import Path

import torch

from workstation_experiments import write_json


ROOT = Path(__file__).resolve().parent
O1_JOB = (ROOT / 'results' / 'summer_o1_session_20261017_232945'
          / 'training' / 'jobs' / 'physical' / 'subject_02' / 'seed_13')


def verify(bridge_directory):
    bridge_directory = Path(bridge_directory).resolve()
    original = json.loads((O1_JOB / 'result.json').read_text())
    bridge = json.loads((bridge_directory / 'result.json').read_text())
    a = torch.load(O1_JOB / 'best.pt', map_location='cpu', weights_only=False)['model']
    b = torch.load(bridge_directory / 'best.pt', map_location='cpu', weights_only=False)['model']
    if a.keys() != b.keys():
        raise ValueError('Model state keys differ')
    max_weight_difference = max(float((a[key].float() - b[key].float()).abs().max())
                                for key in a)
    original_curve = [row['validation_ideal_f1'] for row in original['history']]
    bridge_curve = [row['validation_ideal_f1'] for row in bridge['history']]
    if len(original_curve) != len(bridge_curve):
        raise ValueError('Epoch curve lengths differ')
    max_curve_difference = max(abs(x - y) for x, y in zip(original_curve,
                                                           bridge_curve))
    checks = {
        'same_subject_seed': (original['subject'], original['seed']) ==
                             (bridge['subject'], bridge['seed']) == (2, 13),
        'same_split': original['split'] == bridge['split'],
        'same_epoch_budget': original['epochs'] == bridge['epochs'] == 31,
        'same_best_epoch': original['best_epoch'] == bridge['best_epoch'],
        'weight_tolerance': max_weight_difference <= 1e-4,
        'validation_tolerance': max_curve_difference <= 1e-3,
    }
    report = {'passed': all(checks.values()), 'checks': checks,
              'max_weight_difference': max_weight_difference,
              'max_validation_f1_difference': max_curve_difference,
              'original_job': str(O1_JOB), 'bridge_job': str(bridge_directory),
              'interpretation': 'Checks training parity of raw/physical controls; '
                                'does not evaluate held-out placement superiority.'}
    write_json(bridge_directory.parent / 'bridge_review.json', report)
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bridge', required=True)
    args = parser.parse_args()
    result = verify(args.bridge)
    print(json.dumps(result, indent=2))
    raise SystemExit(0 if result['passed'] else 2)
