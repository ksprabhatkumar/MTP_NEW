"""Recalculate completed O1 placement results after excluding zero-signal files."""
import csv
import json
from pathlib import Path
import numpy as np
from sklearn.metrics import f1_score

ROOT = Path(__file__).resolve().parent
RUN = ROOT / 'results' / 'o1_full_20261016_220301'
OUT = ROOT / 'results' / 'o2_preflight_20261017'
CACHE = ROOT / 'dataset' / 'workstation_o1'


def main():
    OUT.mkdir(exist_ok=True)
    audit = json.loads((CACHE / 'audit.json').read_text())
    excluded = []
    for record in audit['recordings']:
        with np.load(CACHE / (Path(record['source']).stem + '.npz')) as saved:
            all_zero = np.all(saved['x'] == 0, axis=(1,2,3))
            if all_zero.any():
                excluded.append({'source': record['source'], 'subject': record['subject'],
                    'scenario': record['scenario'], 'windows': record['windows'],
                    'zero_windows': int(all_zero.sum()), 'all_zero': bool(all_zero.all())})
    (OUT / 'invalid_recordings.json').write_text(json.dumps(excluded, indent=2))
    print('Invalid recordings', excluded)
    invalid_names = {row['source'] for row in excluded}
    bounds = np.cumsum([0] + [record['windows'] for record in audit['recordings']])
    excluded_corpus_indices = np.concatenate([
        np.arange(bounds[i], bounds[i + 1]) for i, record in enumerate(audit['recordings'])
        if record['source'] in invalid_names])
    rows = []
    for result_path in (RUN / 'jobs').glob('*/*/*/result.json'):
        result = json.loads(result_path.read_text())
        if result['subject'] not in (2, 5, 15):
            continue
        row = {'condition': result['condition'], 'subject': result['subject'],
               'seed': result['seed']}
        for scenario in ('ideal', 'self', 'induced'):
            path = result_path.with_name(f'predictions_{scenario}.npz')
            with np.load(path) as data:
                y, prediction, indices = (data[key] for key in ('labels', 'predictions', 'corpus_indices'))
            keep = ~np.isin(indices, excluded_corpus_indices)
            if not keep.any():
                raise ValueError(f'No valid {scenario} windows for {result_path}')
            row[f'{scenario}_valid_windows'] = int(keep.sum())
            row[f'{scenario}_excluded_windows'] = int((~keep).sum())
            row[f'{scenario}_macro_f1_33'] = float(f1_score(
                y[keep], prediction[keep], labels=list(range(33)), average='macro', zero_division=0))
            row[f'{scenario}_supported_classes'] = len(set(y[keep].tolist()))
            row[f'{scenario}_supported_f1'] = float(f1_score(y[keep], prediction[keep],
                labels=sorted(set(y[keep].tolist())), average='macro', zero_division=0))
        rows.append(row)
    with (OUT / 'corrected_o1_primary.csv').open('w', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    for condition in ('none', 'residual'):
        subset = [r for r in rows if r['condition'] == condition]
        print(condition, 'corrected induced F1', round(np.mean([r['induced_macro_f1_33'] for r in subset]),4),
              'excluded windows per subject15 seed',
              next(r['induced_excluded_windows'] for r in subset if r['subject'] == 15))


if __name__ == '__main__':
    main()
