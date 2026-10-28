"""Audit and cache all REALDISP recordings without overwriting historical tensors."""
import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
CACHE = ROOT / 'dataset' / 'workstation_o1'
STARTS = (28, 41, 15, 54, 2, 93, 80, 106, 67)
COLS = [i for start in STARTS for i in range(start, start + 6)]


def parse_recording(path):
    import re
    m = re.fullmatch(r'subject(\d+)_(ideal|self|mutual|induced)(\d*)\.log', path.name)
    if not m:
        raise ValueError(path.name)
    return int(m[1]), 'induced' if m[2] == 'mutual' else m[2], m[3] or '0'


def process_recording(path_string):
    path = Path(path_string)
    subject, scenario, recording = parse_recording(path)
    out = CACHE / (path.stem + '.npz')
    fingerprint = {'bytes': path.stat().st_size, 'mtime_ns': path.stat().st_mtime_ns, 'version': 1}
    metadata = out.with_suffix('.json')
    if out.exists() and metadata.exists():
        old = json.loads(metadata.read_text())
        if old['fingerprint'] == fingerprint:
            return old
    # Official files are whitespace separated; C parser avoids Python regex overhead.
    frame = pd.read_csv(path, sep=r'\s+', header=None, dtype=np.float32)
    if frame.shape[1] != 120:
        raise ValueError(f'{path.name}: expected 120 columns, got {frame.shape[1]}')
    values = frame.iloc[:, COLS].replace([np.inf, -np.inf], np.nan)
    missing = values.isna().to_numpy().reshape(-1, 9, 6)
    features = values.interpolate(limit_direction='both').fillna(0).to_numpy(dtype=np.float32)
    labels = frame.iloc[:, 119].to_numpy()
    if not np.isfinite(labels).all() or not np.equal(labels, labels.astype(int)).all():
        raise ValueError(f'Invalid activity labels in {path.name}')
    if labels.min() < 0 or labels.max() > 33:
        raise ValueError(f'Out-of-range activity label in {path.name}')
    windows, targets, starts, availability = [], [], [], []
    for start in range(0, len(frame) - 49, 25):
        label = Counter(labels[start:start+50]).most_common(1)[0][0]
        if label == 0:
            continue
        windows.append(features[start:start+50].reshape(50, 9, 6))
        targets.append(int(label) - 1)
        starts.append(start)
        availability.append(~missing[start:start+50].all(axis=(0, 2)))
    x = np.asarray(windows, dtype=np.float32)
    if not len(x) or not np.isfinite(x).all():
        raise ValueError(f'Empty/nonfinite recording: {path.name}')
    a = np.asarray(availability, dtype=bool)
    x *= a[:, None, :, None]
    np.savez(out, x=x, y=np.asarray(targets, dtype=np.int64),
             starts=np.asarray(starts, dtype=np.int64), availability=a)
    info = {'source': path.name, 'subject': subject, 'scenario': scenario,
            'recording': recording, 'rows': len(frame), 'windows': len(x),
            'class_counts': dict(Counter(targets)), 'missing_values': int(missing.sum()),
            'unavailable_sensor_windows': int((~a).sum()), 'fingerprint': fingerprint}
    metadata.write_text(json.dumps(info, indent=2))
    return info


def prepare(workers=4):
    CACHE.mkdir(parents=True, exist_ok=True)
    paths = sorted((ROOT / 'har_data' / 'REALDISP').glob('subject*.log'))
    # C-parser threads avoid Windows named-pipe restrictions on process pools.
    with ThreadPoolExecutor(max_workers=workers) as pool:
        records = []
        for info in pool.map(process_recording, map(str, paths)):
            records.append(info)
            print(f"Prepared {info['source']}: {info['windows']} windows", flush=True)
    ideal_subjects = sorted({r['subject'] for r in records if r['scenario'] == 'ideal'})
    induced_subjects = sorted({r['subject'] for r in records if r['scenario'] == 'induced'})
    if ideal_subjects != list(range(1, 18)) or induced_subjects != [2, 5, 15]:
        raise ValueError(f'Unexpected coverage: ideal={ideal_subjects}, induced={induced_subjects}')
    audit = {'schema_version': 1, 'shape': [50, 9, 6], 'stride': 25,
             'node_column_starts': STARTS, 'normalization': 'none; fitted per training fold',
             'ideal_subjects': ideal_subjects, 'induced_subjects': induced_subjects,
             'total_windows': sum(r['windows'] for r in records), 'recordings': records}
    (CACHE / 'audit.json').write_text(json.dumps(audit, indent=2))
    print(f"Audit complete: {len(records)} recordings, {audit['total_windows']} windows", flush=True)
    return audit


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--workers', type=int, default=4)
    prepare(parser.parse_args().workers)
