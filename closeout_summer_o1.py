"""Verify the completed O1 training and record an analysis-only plot repair."""
from datetime import datetime
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent
RUN = ROOT / 'results' / 'summer_o1_session_20261017_232945'


def atomic_json(path, obj):
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(obj, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    temp.replace(path)


def main():
    training = RUN / 'training'
    results = list((training / 'jobs').glob('*/*/*/result.json'))
    if len(results) != 126:
        raise RuntimeError(f'Expected 126 finished jobs, found {len(results)}')
    if json.loads((training / 'completed.json').read_text())['completed_jobs'] != 126:
        raise RuntimeError('Training completion marker differs from job count')
    for result in results:
        data = json.loads(result.read_text(encoding='utf-8'))
        if not all(key in data for key in ('condition', 'subject', 'seed', 'metrics')):
            raise RuntimeError(f'Incomplete job record: {result}')
    required = [training / 'ANALYSIS.md', training / 'analysis_summary.json',
                training / 'figures' / 'placement.png']
    if any(not path.is_file() or path.stat().st_size == 0 for path in required):
        raise RuntimeError('The final O1 analysis artifacts are incomplete')
    status_file = RUN / 'status.json'
    original = json.loads(status_file.read_text(encoding='utf-8'))
    if original['state'] not in ('failed', 'complete'):
        raise RuntimeError(f'Unexpected outer state: {original["state"]}')
    completed_file = RUN / 'completed.json'
    completed = json.loads(completed_file.read_text(encoding='utf-8'))
    if set(completed) != {'preflight', 'smoke', 'source_pilot', 'training'} and not (
            original['state'] == 'complete' and 'analysis' in completed):
        raise RuntimeError('Prior supervisor stages are not complete')
    if original['state'] == 'complete':
        print('O1 is already marked complete; verified 126 jobs and analysis outputs')
        return
    if original.get('task') != 'analysis':
        raise RuntimeError('The previous failure was not the final analysis task')
    now = datetime.now().astimezone().isoformat()
    backup = RUN / 'status_before_analysis_repair.json'
    if not backup.exists():
        atomic_json(backup, original)
    evidence = {
        'method': 'analysis-only recovery after incompatible inherited PYTHONPATH',
        'command': 'C:/ProgramData/anaconda3/python.exe analyze_summer_o1.py --run '
                   'results/summer_o1_session_20261017_232945/training',
        'job_records_verified': 126,
        'artifacts': {str(path.relative_to(RUN)): hashlib.sha256(path.read_bytes()).hexdigest()
                      for path in required},
        'original_status': backup.name,
        'verified_at': now,
        'verifier': sys.executable,
    }
    atomic_json(RUN / 'analysis_repair.json', evidence)
    completed['analysis'] = {'started': now, 'finished': now, 'plan_step': 'P4.7',
                             'method': 'analysis_only_repair',
                             'evidence': 'analysis_repair.json'}
    atomic_json(completed_file, completed)
    status = {key: value for key, value in original.items()
              if key not in ('error', 'traceback', 'child_pid', 'command')}
    status.update(state='complete', completed=5, total=5, task='analysis',
                  plan_step='P4.7', updated=now, repair='analysis_repair.json')
    atomic_json(status_file, status)
    print(json.dumps({'state': status['state'], 'jobs': len(results),
                      'placement_figure': str(required[-1]), 'repair': str(RUN / 'analysis_repair.json')}))


if __name__ == '__main__':
    main()
