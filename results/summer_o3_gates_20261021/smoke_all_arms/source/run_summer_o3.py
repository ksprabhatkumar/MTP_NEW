"""Staged, resumable subject-client Objective 3 experiment runner."""

import argparse
from dataclasses import asdict
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import threading
import traceback

ROOT = Path(__file__).resolve().parent
SOURCES = ['run_summer_o3.py', 'summer_o3_experiments.py',
    'summer_o3_federation.py', 'prepare_o3_clients.py',
    'check_summer_o3_cpu.py', 'summer_o2_experiments.py',
    'models/summer_dynamic.py', 'o2_features.py', 'o2_experiments.py',
    'summer_o1_experiments.py', 'workstation_experiments.py',
    'run_training.py']


def selection_gate(config):
    problems = []
    if config.get('selected_invariant') not in ('magnitude', 'mag_dot'):
        problems.append('Select magnitude or mag_dot from the source-validation evidence')
    o2 = config.get('summer_o2_run')
    if not o2:
        problems.append('Set summer_o2_run to the completed summer-based O2 session')
    else:
        o2 = Path(o2).resolve()
        status = o2 / 'status.json'
        analysis = o2 / 'training' / 'ANALYSIS.md'
        completed = o2 / 'training' / 'completed.json'
        if not status.exists() or json.loads(status.read_text()).get('state') != 'complete':
            problems.append('Summer-based O2 supervisor is incomplete')
        if not analysis.is_file():
            problems.append('Summer-based O2 analysis is missing')
        if not completed.is_file() or json.loads(completed.read_text()).get('completed_jobs') != 63:
            problems.append('Summer-based O2 has fewer than 63 new jobs')
    return problems


def jobs_for(config, methods):
    from summer_o3_experiments import O3Job
    invariant = config['selected_invariant']
    arms = [('raw', 'residual'), (invariant, 'physical'),
            (invariant, 'residual'), (invariant, 'none')]
    return [O3Job(kind, graph, method, subject, seed)
            for subject in (2, 5, 15) for kind, graph in arms
            for method in methods for seed in (13, 29, 47)]


def publish(output, data):
    from workstation_experiments import write_json
    state = dict(data, pid=os.getpid(), updated=datetime.now().astimezone().isoformat())
    write_json(output / 'status.json', state)
    (output / 'TRAINING_HANDOFF.txt').write_text(
        'SUMMER OBJECTIVE 3 SUBJECT-CLIENT SESSION\n'
        + json.dumps(state, indent=2) + '\n\n'
        + 'Each source subject is one client. Held-out test and validation subjects never train.\n'
        + 'Inspect the selected O2 feature config, round logs, client metrics, '
          'communication bytes and checkpoints. Resume only in the same output '
          'directory after the prior writer has exited.\n', encoding='utf-8')


def summary_for(output, jobs):
    from workstation_experiments import write_json
    rows = [json.loads(path.read_text()) for path in
            (output / 'jobs').glob('*/*/*/*/result.json')]
    by_method = {}
    for method in ('centralized', 'local_only', 'fedavg', 'fedbn'):
        selected = [row for row in rows if row['job']['method'] == method]
        by_method[method] = {'completed': len(selected),
            'expected': sum(job.method == method for job in jobs),
            'communication_bytes': sum(row.get('communication_bytes', 0)
                                       for row in selected)}
    write_json(output / 'summary.json', {'completed_jobs': len(rows),
        'expected_jobs': len(jobs), 'methods': by_method})


def run(args, config):
    import msvcrt
    import torch
    from o2_features import O2Corpus
    from summer_o3_experiments import METHODS, run_job, split_clients
    from workstation_experiments import write_json
    from run_training import telemetry, Tee

    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    lock = (output / '.run.lock').open('a+b')
    if lock.tell() == 0:
        lock.write(b'0'); lock.flush()
    lock.seek(0)
    msvcrt.locking(lock.fileno(), msvcrt.LK_NBLCK, 1)
    log = (output / 'run.log').open('a', encoding='utf-8', buffering=1)
    sys.stdout, sys.stderr = Tee(sys.stdout, log), Tee(sys.stderr, log)
    stop = threading.Event()
    monitor = None
    try:
        problems = selection_gate(config)
        if problems:
            raise RuntimeError('O3 entry gate:\n- ' + '\n- '.join(problems))
        if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
            raise RuntimeError('CUDA BF16 GPU required')
        torch.set_num_threads(8)
        torch.set_num_interop_threads(2)
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        print('SUMMER O3', args.mode, output, torch.cuda.get_device_name(), flush=True)
        hashes = {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
                  for name in SOURCES}
        fingerprint = {'sources': hashes,
            'config': hashlib.sha256(Path(args.config).read_bytes()).hexdigest(),
            'cache_audit': hashlib.sha256(
                (ROOT / 'dataset/workstation_o1/audit.json').read_bytes()).hexdigest()}
        if (output / 'fingerprint.json').exists() and json.loads(
                (output / 'fingerprint.json').read_text()) != fingerprint:
            raise ValueError('O3 source, config or cache changed')
        write_json(output / 'fingerprint.json', fingerprint)
        if not (output / 'source').exists():
            for name in SOURCES:
                destination = output / 'source' / name
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(ROOT / name, destination)
            shutil.copy2(args.config, output / 'o3_study_at_start.json')
        write_json(output / 'system.json', {'python': sys.version,
            'executable': sys.executable, 'torch': torch.__version__,
            'cuda': torch.version.cuda, 'gpu': torch.cuda.get_device_name()})
        with (output / 'requirements.lock.txt').open('w') as req:
            subprocess.run([sys.executable, '-m', 'pip', 'freeze'],
                           stdout=req, check=True)
        monitor = threading.Thread(target=telemetry, args=(output, stop),
                                   daemon=True)
        monitor.start()
        corpus = O2Corpus()
        invariant = config['selected_invariant']
        manifest = json.loads((ROOT / config['client_manifest']).read_text())
        folds = {}
        for subject in (2, 5, 15):
            fold = corpus.fold(subject, invariant)
            clients = split_clients(corpus, fold)
            expected = manifest['folds'][str(subject)]
            if len(clients) != expected['client_count'] or sum(
                    len(part['y']) for part in clients.values()) != expected['training_windows']:
                raise ValueError('O3 client manifest/data mismatch')
            folds[str(subject)] = {'split': fold['summary'],
                                   'client_counts': {str(key): len(value['y'])
                                                     for key, value in clients.items()}}
            del fold, clients
            torch.cuda.empty_cache()
        write_json(output / 'fold_preflight.json', folds)
        if args.mode == 'preflight':
            publish(output, {'state': 'complete', 'plan_step': 'P6.1'})
            write_json(output / 'passed.json', {'passed': True,
                'client_count_each_fold': 15, 'invariant': invariant})
        elif args.mode == 'smoke':
            publish(output, {'state': 'smoke', 'plan_step': 'P6.3'})
            fold = corpus.fold(2, invariant)
            from summer_o3_experiments import O3Job
            smoke_jobs = [(invariant, 'residual', method) for method in METHODS]
            smoke_jobs += [(invariant, 'physical', 'centralized'),
                           (invariant, 'none', 'centralized'),
                           (invariant, 'none', 'fedavg')]
            for kind, graph, method in smoke_jobs:
                job = O3Job(kind, graph, method, 2, 13)
                run_job(corpus, fold, job, 2, 64,
                    output / 'jobs' / job.arm / method, smoke_limit=2)
            write_json(output / 'passed.json', {'passed': True,
                'jobs': smoke_jobs, 'research_result': False})
            publish(output, {'state': 'complete', 'plan_step': 'P6.3'})
        else:
            methods = tuple(args.methods.split(','))
            if not methods or any(method not in METHODS for method in methods):
                raise ValueError('Invalid O3 method list')
            jobs = jobs_for(config, methods)
            schedule = {'jobs': [asdict(job) for job in jobs],
                'epochs_or_rounds': 31, 'batch_size': 64,
                'methods': methods, 'client_participation': 'all 15 each round',
                'selection': 'source ideal macro-F1 only'}
            schedule_file = output / 'schedule.json'
            if schedule_file.exists() and json.loads(
                    schedule_file.read_text()) != json.loads(json.dumps(schedule)):
                raise ValueError('O3 schedule changed')
            write_json(schedule_file, schedule)
            for number, job in enumerate(jobs, 1):
                destination = output / 'jobs' / job.arm / job.method / f'subject_{job.subject:02d}' / f'seed_{job.seed}'
                if (destination / 'result.json').exists():
                    continue
                fold = corpus.fold(job.subject, job.kind)
                publish(output, {'state': 'training', 'plan_step': 'P6.5',
                    'current_job': number, 'total_jobs': len(jobs),
                    'job': asdict(job)})
                run_job(corpus, fold, job, 31, 64, destination)
                summary_for(output, jobs)
                del fold
                torch.cuda.empty_cache()
            write_json(output / 'completed.json', {'completed_jobs': len(jobs),
                'finished': datetime.now().astimezone().isoformat()})
            publish(output, {'state': 'complete', 'plan_step': 'P6.5'})
    except BaseException as error:
        publish(output, {'state': 'failed', 'error': repr(error),
            'traceback': traceback.format_exc()})
        raise
    finally:
        stop.set()
        if monitor:
            monitor.join(timeout=12)
        lock.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode', choices=('plan', 'preflight', 'smoke', 'full'),
                        required=True)
    parser.add_argument('--config', default='summer_o3_study.json')
    parser.add_argument('--output')
    parser.add_argument('--methods', default='centralized,local_only,fedavg,fedbn')
    args = parser.parse_args()
    config = json.loads(Path(args.config).read_text())
    if args.mode == 'plan':
        print(json.dumps({'ready': not selection_gate(config),
            'blockers': selection_gate(config), 'provisional_bundles': 144,
            'gpu_used': False}, indent=2))
    else:
        if not args.output:
            parser.error('--output is required for GPU modes')
        run(args, config)
