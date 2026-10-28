"""Freeze O3's invariant feature from O2 source-validation scores only."""

import argparse
import json
from pathlib import Path
import statistics

import torch

from workstation_experiments import write_json


ROOT = Path(__file__).resolve().parent


def finalize(o2_run):
    o2_run = Path(o2_run).resolve()
    completed = json.loads((o2_run / 'training' / 'completed.json').read_text())
    if completed.get('completed_jobs') != 63:
        raise ValueError('Summer O2 training is not complete')
    if not (o2_run / 'training' / 'ANALYSIS.md').is_file():
        raise ValueError('Summer O2 analysis is missing')
    scores = {}
    for kind in ('magnitude', 'mag_dot'):
        files = list((o2_run / 'training' / 'jobs' / f'{kind}_residual')
                     .glob('*/*/result.json'))
        keys = {(int(path.parent.parent.name.removeprefix('subject_')),
                 int(path.parent.name.removeprefix('seed_'))) for path in files}
        expected = {(subject, seed) for subject in (2, 5, 15)
                    for seed in (13, 29, 47)}
        if keys != expected or len(files) != 9:
            raise ValueError(f'Incomplete source-validation scores for {kind}')
        # Read checkpoint metadata only; result.json contains held-out scores.
        scores[kind] = statistics.mean(torch.load(path.parent / 'best.pt',
            map_location='cpu', weights_only=False)['validation_f1']
            for path in files)
    selected = ('mag_dot' if scores['mag_dot'] >= scores['magnitude']
                else 'magnitude')
    evidence = {'selection_rule': 'mean best source-ideal 33-class macro-F1 '
                'of nine residual runs; exact tie -> mag_dot',
                'scores': scores, 'selected_invariant': selected,
                'held_out_placement_read_for_selection': False,
                'summer_o2_run': str(o2_run)}
    study_file = ROOT / 'summer_o3_study.json'
    study = json.loads(study_file.read_text())
    if study.get('selected_invariant') not in (None, selected):
        raise ValueError('Existing O3 invariant selection conflicts')
    study['selected_invariant'] = selected
    study['summer_o2_run'] = str(o2_run)
    study['status'] = 'ready_for_o3_preflight_and_smoke'
    study['selection_evidence'] = str(o2_run / 'o3_selection.json')
    write_json(study_file, study)
    plan_file = ROOT / 'summer_o3_session.json'
    plan = json.loads(plan_file.read_text())
    plan['ready'] = True
    plan['unresolved'] = []
    write_json(plan_file, plan)
    write_json(o2_run / 'o3_selection.json', evidence)
    return evidence


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--o2-run', required=True)
    args = parser.parse_args()
    print(json.dumps(finalize(args.o2_run), indent=2))
