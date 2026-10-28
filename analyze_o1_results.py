"""Post-hoc, read-only analysis of saved O1 predictions and fold results.

Writes only new analysis artifacts inside the run directory. Never selects
checkpoints or changes the frozen experiments.
"""
import argparse
import csv
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import accuracy_score, f1_score


def write_csv(path, rows):
    if not rows:
        return
    with path.open('w', newline='', encoding='utf-8') as file:
        writer = csv.DictWriter(file, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)


def analyze(root):
    root = Path(root)
    audit = json.loads((root / 'data_audit.json').read_text())
    records = audit['recordings']
    bounds = np.cumsum([0] + [record['windows'] for record in records])
    fold_rows = list(csv.DictReader((root / 'fold_metrics.csv').open(encoding='utf-8')))
    assert len(fold_rows) == 118
    results = [json.loads(path.read_text()) for path in (root / 'jobs').glob('*/*/*/result.json')]
    assert len(results) == 118
    result_map = {(r['condition'], r['subject'], r['seed']): r for r in results}
    primary = [r for r in results if r['subject'] in (2, 5, 15)]
    names = sorted({r['condition'] for r in primary})
    overview = []
    for name in names:
        group = [r for r in primary if r['condition'] == name]
        row = {'condition': name, 'jobs': len(group)}
        for scenario in ('ideal', 'self', 'induced'):
            row[f'{scenario}_macro_f1_33'] = np.mean([r['metrics'][scenario]['macro_f1'] for r in group])
            row[f'{scenario}_accuracy'] = np.mean([r['metrics'][scenario]['accuracy'] for r in group])
        row['matched_ideal_to_induced_drop'] = np.mean(
            [r['matched_placement']['degradation_macro_f1'] for r in group])
        row['median_batch_one_ms'] = np.median([r['efficiency']['batch_one_median_ms'] for r in group])
        row['parameters'] = group[0]['efficiency']['parameters']
        overview.append(row)
    write_csv(root / 'analysis_overview.csv', overview)

    pairs = []
    for subject in (2, 5, 15):
        for seed in (13, 29, 47):
            base = result_map['none', subject, seed]
            treatment = result_map['residual', subject, seed]
            pairs.append({'subject': subject, 'seed': seed,
                **{f'delta_residual_minus_none_{scenario}_f1':
                    treatment['metrics'][scenario]['macro_f1'] - base['metrics'][scenario]['macro_f1']
                    for scenario in ('ideal', 'self', 'induced')},
                'delta_residual_minus_none_matched_drop':
                    treatment['matched_placement']['degradation_macro_f1'] -
                    base['matched_placement']['degradation_macro_f1']})
    write_csv(root / 'analysis_paired_seed_differences.csv', pairs)

    # Predictions retain corpus indices. Map each index to the raw recording,
    # then evaluate both models on identical subject/recording/class supports.
    recording_rows = []
    for subject in (2, 5, 15):
        for seed in (13, 29, 47):
            paths = {name: root / 'jobs' / name / f'subject_{subject:02d}' / f'seed_{seed}'
                     for name in ('none', 'residual')}
            arrays = {}
            for name, path in paths.items():
                arrays[name] = {}
                for scenario in ('ideal', 'induced'):
                    with np.load(path / f'predictions_{scenario}.npz') as archive:
                        arrays[name][scenario] = {key: archive[key] for key in archive.files}
            ideal_labels = arrays['none']['ideal']['labels']
            ideal_predictions = {name: arrays[name]['ideal']['predictions'] for name in paths}
            for record_id, record in enumerate(records):
                if record['subject'] != subject or record['scenario'] != 'induced':
                    continue
                induced = arrays['none']['induced']
                selected = ((induced['corpus_indices'] >= bounds[record_id]) &
                            (induced['corpus_indices'] < bounds[record_id + 1]))
                y = induced['labels'][selected]
                supported = sorted(set(ideal_labels.tolist()) & set(y.tolist()))
                ideal_selected = np.isin(ideal_labels, supported)
                row = {'subject': subject, 'seed': seed, 'recording': record['source'],
                       'displaced_sensors': int(record['recording']),
                       'windows': len(y), 'present_classes': len(set(y.tolist())),
                       'matched_classes': len(supported)}
                for name in paths:
                    prediction = arrays[name]['induced']['predictions'][selected]
                    kept = np.isin(y, supported)
                    row[f'{name}_all33_f1'] = f1_score(y, prediction, labels=list(range(33)),
                                                        average='macro', zero_division=0)
                    row[f'{name}_matched_f1'] = f1_score(y[kept], prediction[kept],
                                labels=supported, average='macro', zero_division=0)
                    row[f'{name}_accuracy'] = accuracy_score(y, prediction)
                    row[f'{name}_ideal_same_classes_f1'] = f1_score(
                        ideal_labels[ideal_selected], ideal_predictions[name][ideal_selected],
                        labels=supported, average='macro', zero_division=0)
                row['residual_minus_none_matched_f1'] = row['residual_matched_f1'] - row['none_matched_f1']
                recording_rows.append(row)
    write_csv(root / 'analysis_displacement_recordings.csv', recording_rows)

    ideal_loso = []
    for subject in range(1, 18):
        control = result_map['none', subject, 13]
        treatment = result_map['residual', subject, 13]
        classes = sum(item['support'] > 0 for item in control['metrics']['ideal']['per_class'])
        ideal_loso.append({'subject': subject, 'ideal_windows': control['metrics']['ideal']['windows'],
            'present_classes': classes, 'none_macro_f1_33': control['metrics']['ideal']['macro_f1'],
            'residual_macro_f1_33': treatment['metrics']['ideal']['macro_f1'],
            'residual_minus_none_f1': treatment['metrics']['ideal']['macro_f1'] -
                                      control['metrics']['ideal']['macro_f1'],
            'none_supported_class_f1': np.mean([item['f1'] for item in control['metrics']['ideal']['per_class']
                                                if item['support'] > 0]),
            'residual_supported_class_f1': np.mean([item['f1'] for item in treatment['metrics']['ideal']['per_class']
                                                    if item['support'] > 0])})
    write_csv(root / 'analysis_ideal_loso_17_subjects.csv', ideal_loso)
    print(f'Analyzed {len(results)} jobs, {len(pairs)} paired runs, {len(recording_rows)} recording-level tests.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run_directory')
    analyze(parser.parse_args().run_directory)
