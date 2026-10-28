"""CPU-only subject-client manifests matching frozen summer O1/O2 folds."""

import argparse
import json
from pathlib import Path

from workstation_experiments import write_json


ROOT = Path(__file__).resolve().parent
O1 = ROOT / 'results' / 'summer_o1_session_20261017_232945' / 'training'
AUDIT = ROOT / 'dataset' / 'workstation_o1' / 'audit.json'
INVALID = {'subject6_self.log', 'subject13_self.log',
           'subject15_mutual4.log'}


def collect(rows, subject, scenario):
    selected = [row for row in rows if row['subject'] == subject
                and row['scenario'] == scenario and row['source'] not in INVALID]
    count = sum(row['windows'] for row in selected)
    classes = sorted({int(label) for row in selected
                      for label, amount in row['class_counts'].items()
                      if amount > 0})
    return {'recordings': [row['source'] for row in selected],
            'windows': count, 'classes': classes,
            'class_count': len(classes)}


def build():
    audit = json.loads(AUDIT.read_text())
    frozen = json.loads((O1 / 'fold_preflight.json').read_text())
    rows = audit['recordings']
    result = {'source': str(AUDIT), 'frozen_folds': str(O1 / 'fold_preflight.json'),
              'excluded_recordings': sorted(INVALID), 'folds': {}}
    for test in (2, 5, 15):
        fold = frozen[str(test)]
        validation = fold['validation_subject']
        subjects = fold['train_subjects']
        if len(subjects) != 15 or set(subjects) != set(range(1, 18)) - {test, validation}:
            raise ValueError(f'Unexpected training client list for subject {test}')
        clients = {}
        for subject in subjects:
            ideal = collect(rows, subject, 'ideal')
            if ideal['windows'] <= 0:
                raise ValueError(f'Empty ideal client {subject}')
            clients[str(subject)] = {'subject': subject, 'train_ideal': ideal,
                'known_self': collect(rows, subject, 'self'),
                'known_induced': collect(rows, subject, 'induced')}
        train_count = sum(client['train_ideal']['windows']
                          for client in clients.values())
        if train_count != fold['counts']['train']:
            raise ValueError(f'Train count mismatch for fold {test}')
        target = {scenario: collect(rows, test, scenario)
                  for scenario in ('ideal', 'self', 'induced')}
        for scenario in ('ideal', 'self', 'induced'):
            if target[scenario]['windows'] != fold['counts'][scenario]:
                raise ValueError(f'Test {scenario} count mismatch for fold {test}')
        source_validation = {scenario: collect(rows, validation, scenario)
                             for scenario in ('ideal', 'self')}
        for scenario in ('ideal', 'self'):
            if source_validation[scenario]['windows'] != fold['counts']['val_' + scenario]:
                raise ValueError(f'Validation {scenario} count mismatch for fold {test}')
        result['folds'][str(test)] = {'test_subject': test,
            'validation_subject': validation, 'clients': clients,
            'training_windows': train_count, 'validation': source_validation,
            'unseen_test': target, 'client_count': len(clients),
            'thin_clients': [client['subject'] for client in clients.values()
                if client['train_ideal']['class_count'] < 30]}
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    manifest = build()
    write_json(Path(args.output), manifest)
    print(json.dumps({'folds': {key: {
        'clients': value['client_count'],
        'training_windows': value['training_windows'],
        'thin_clients': value['thin_clients']}
        for key, value in manifest['folds'].items()}}, indent=2))
