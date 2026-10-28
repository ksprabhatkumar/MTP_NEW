"""Read-only raw REALDISP audit against the frozen O1 cache."""
import csv
import json
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
CACHE = ROOT / 'dataset' / 'workstation_o1'


def main(output):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    audit = json.loads((CACHE / 'audit.json').read_text())
    rows = []
    for record in audit['recordings']:
        path = ROOT / 'har_data' / 'REALDISP' / record['source']
        if path.stat().st_size != record['fingerprint']['bytes']:
            raise ValueError(f'Raw file changed: {path.name}')
        frame = pd.read_csv(path, sep=r'\s+', header=None, usecols=[0, 1, 119], dtype=np.float64)
        sec, micro, labels = (frame.iloc[:, i].to_numpy() for i in range(3))
        stamps = sec + micro * 1e-6
        delta = np.diff(stamps)
        if not np.isfinite(stamps).all() or not np.isfinite(labels).all():
            raise ValueError(f'Nonfinite timestamp or label: {path.name}')
        if np.min(labels) < 0 or np.max(labels) > 33 or not np.equal(labels, labels.astype(int)).all():
            raise ValueError(f'Invalid labels: {path.name}')
        raw_classes = sorted({int(v) - 1 for v in labels if v > 0})
        cached_classes = sorted(int(v) for v in record['class_counts'])
        if not set(cached_classes).issubset(raw_classes):
            raise ValueError(f'Cached labels not present in raw file: {path.name}')
        with np.load(CACHE / (path.stem + '.npz')) as archive:
            y, starts, x = archive['y'], archive['starts'], archive['x']
            if len(y) != record['windows'] or x.shape != (len(y), 50, 9, 6):
                raise ValueError(f'Window mismatch: {path.name}')
            if not np.isfinite(x).all():
                raise ValueError(f'Nonfinite cached features: {path.name}')
            # Independent check of the majority-label contract, without
            # assuming all raw activities survive the 50-sample window rule.
            for index in np.linspace(0, len(y) - 1, min(16, len(y)), dtype=int):
                raw = labels[starts[index]:starts[index] + 50].astype(int)
                winner = int(np.bincount(raw, minlength=34).argmax()) - 1
                if winner != y[index]:
                    # Counter.most_common uses first-occurrence tie breaking.
                    counts = np.bincount(raw, minlength=34)
                    if counts.max() != counts[y[index] + 1]:
                        raise ValueError(f'Window/label mismatch: {path.name} at {starts[index]}')
        row = {'source': path.name, 'subject': record['subject'], 'scenario': record['scenario'],
               'recording': record['recording'], 'raw_rows': len(labels),
               'raw_classes': len(raw_classes), 'window_classes': len(cached_classes),
               'windows': len(y), 'null_rows': int((labels == 0).sum()),
               'invalid_or_reverse_time_steps': int((delta <= 0).sum()),
               'median_sample_interval_seconds': float(np.median(delta)),
               'p99_sample_interval_seconds': float(np.percentile(delta, 99)),
               'missing_feature_values': record['missing_values']}
        rows.append(row)
        print(path.name, 'raw classes', row['raw_classes'], 'windows', row['windows'],
              'time anomalies', row['invalid_or_reverse_time_steps'], flush=True)
    with (output / 'raw_recordings.csv').open('w', newline='', encoding='utf-8') as file:
        writer = csv.DictWriter(file, fieldnames=rows[0])
        writer.writeheader()
        writer.writerows(rows)
    summary = {'files': len(rows), 'raw_rows': sum(row['raw_rows'] for row in rows),
               'windows': sum(row['windows'] for row in rows),
               'nonmonotone_time_steps': sum(row['invalid_or_reverse_time_steps'] for row in rows),
               'suspicious': [row for row in rows if row['invalid_or_reverse_time_steps'] or
                              row['window_classes'] < 29]}
    (output / 'raw_summary.json').write_text(json.dumps(summary, indent=2))
    print('AUDIT COMPLETE', json.dumps({k: v for k, v in summary.items() if k != 'suspicious'}))


if __name__ == '__main__':
    main(ROOT / 'results' / 'o2_preflight_20261017')
