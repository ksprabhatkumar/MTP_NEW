"""One entry point: audit, test, smoke, calibrate, then sequential O1 training.

    prabhat\Scripts\python.exe -u run_training.py --mode full --hours 5.5
    prabhat\Scripts\python.exe -u run_training.py --mode smoke
    prabhat\Scripts\python.exe -u run_training.py --mode full --output results/RUN
"""
import argparse
from contextlib import contextmanager
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import threading
import time
import traceback

ROOT = Path(__file__).resolve().parent
os.chdir(ROOT)


class Tee:
    def __init__(self, *streams):
        self.streams = streams

    def write(self, text):
        for stream in self.streams:
            stream.write(text)
            stream.flush()

    def flush(self):
        for stream in self.streams:
            stream.flush()


def telemetry(output, stop):
    with (output / 'gpu_telemetry.csv').open('a', buffering=1) as handle:
        while not stop.is_set():
            try:
                row = subprocess.check_output(['nvidia-smi',
                    '--query-gpu=timestamp,utilization.gpu,utilization.memory,memory.used,memory.total,temperature.gpu,power.draw',
                    '--format=csv,noheader,nounits'], text=True,
                    creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0), timeout=10)
                handle.write(row)
            except Exception as error:
                print(f'Telemetry unavailable: {error}', flush=True)
            stop.wait(15)


def run(args):
    import numpy as np
    import torch
    from prepare_workstation_data import CACHE, prepare
    from workstation_experiments import (CONDITIONS, Corpus, make_model, optimizer_for,
        seed_all, train_epoch, evaluate, train_job, summarize, write_json)
    output = Path(args.output).resolve() if args.output else ROOT / 'results' / (
        datetime.now().strftime('%Y%m%d_%H%M%S') + '_o1_' + args.mode)
    output.mkdir(parents=True, exist_ok=True)
    log = (output / 'run.log').open('a', encoding='utf-8', buffering=1)
    sys.stdout, sys.stderr = Tee(sys.stdout, log), Tee(sys.stderr, log)
    print(f'OUTPUT: {output}', flush=True)
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError('This workstation suite requires CUDA with bfloat16 support; no silent CPU fallback.')
    torch.set_num_threads(min(8, os.cpu_count() or 1))
    torch.set_num_interop_threads(2)
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    properties = torch.cuda.get_device_properties(0)
    system = {'python': sys.version, 'executable': sys.executable, 'platform': platform.platform(),
              'logical_cpus': os.cpu_count(), 'torch': torch.__version__, 'cuda': torch.version.cuda,
              'gpu': properties.name, 'gpu_bytes': properties.total_memory,
              'batch_size': args.batch_size, 'cpu_threads': torch.get_num_threads(),
              'bf16': True, 'tf32': True, 'cuda_data_resident': True,
              'determinism': 'fixed seeds; optimized CUDA kernels may remain nondeterministic',
              'environment_installation': 'offline copy of compatible local Python 3.11 distributions; pip check passed'}
    write_json(output / 'system.json', system)
    subprocess.run([sys.executable, '-m', 'pip', 'freeze'], stdout=(output / 'requirements.lock.txt').open('w'), check=True)
    snapshot = output / 'source'
    if not snapshot.exists():
        snapshot.mkdir()
        for path in [ROOT / name for name in ('run_training.py', 'workstation_experiments.py',
                'prepare_workstation_data.py', 'TRAINING_PLAN.md', 'requirements.txt')] + list((ROOT / 'models').glob('*.py')) + [ROOT / 'baselines' / 'dynamicwhar_adapter.py']:
            target = snapshot / path.relative_to(ROOT)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, target)
        shutil.copytree(ROOT / 'baselines' / 'DynamicWHAR', snapshot / 'baselines' / 'DynamicWHAR',
                        ignore=shutil.ignore_patterns('.git', '__pycache__'))
    hashes = {str(p.relative_to(snapshot)): hashlib.sha256(p.read_bytes()).hexdigest()
              for p in snapshot.rglob('*') if p.is_file()}
    write_json(output / 'source_hashes.json', hashes)
    stop = threading.Event()
    monitor = threading.Thread(target=telemetry, args=(output, stop), daemon=True)
    monitor.start()
    status_path = output / 'status.json'
    try:
        write_json(status_path, {'state': 'preflight', 'pid': os.getpid(), 'started': datetime.now().isoformat()})
        if args.rebuild_data or not (CACHE / 'audit.json').exists():
            prepare(args.workers)
        shutil.copy2(CACHE / 'audit.json', output / 'data_audit.json')
        tests = subprocess.run([sys.executable, '-m', 'unittest', 'discover', '-s', 'tests', '-v'],
                               capture_output=True, text=True)
        (output / 'unit_tests.log').write_text(tests.stdout + tests.stderr)
        if tests.returncode:
            raise RuntimeError('Preflight unit tests failed; see unit_tests.log')
        corpus = Corpus()
        fold = corpus.fold(2)
        print(f'Full corpus loaded: {len(corpus.y)} windows; GPU-resident fold: {fold["summary"]["counts"]}', flush=True)
        smoke_done = output / 'smoke' / 'passed.json'
        if not smoke_done.exists():
            for condition in CONDITIONS:
                train_job(fold, condition, 2, 13, 2, args.batch_size,
                          output / 'smoke' / condition.name, limit=2)
            write_json(smoke_done, {'passed': True, 'conditions': [c.name for c in CONDITIONS],
                                   'epochs': 2, 'batch_limit': 2, 'research_result': False})
        if args.mode == 'smoke':
            write_json(status_path, {'state': 'smoke_complete', 'pid': os.getpid(),
                                     'finished': datetime.now().isoformat()})
            print('All smoke conditions passed.', flush=True)
            return
        # 90 primary jobs plus 28 additional seed-13 ideal LOSO folds.
        jobs = []
        for subject in [2, 5, 15] + [s for s in range(1, 18) if s not in (2, 5, 15)]:
            conditions = CONDITIONS if subject in (2, 5, 15) else [c for c in CONDITIONS if c.name in ('none', 'residual')]
            seeds = [13, 29, 47] if subject in (2, 5, 15) else [13]
            jobs.extend({'subject': subject, 'condition': c.name, 'seed': seed} for c in conditions for seed in seeds)
        plan_path = output / 'schedule.json'
        if plan_path.exists():
            plan = json.loads(plan_path.read_text())
            if plan['batch_size'] != args.batch_size:
                raise ValueError('Resume must use the original batch size')
            epochs, jobs = plan['epochs'], plan['jobs']
        else:
            calibration = {}
            print('Calibrating all conditions on full training/validation data; no test selection.', flush=True)
            for condition in CONDITIONS:
                seed_all(101)
                model = make_model(condition)
                optimizer = optimizer_for(model)
                # Warm kernels then time a complete real-data epoch and validation.
                train_epoch(model, optimizer, fold['data']['train'], condition, args.batch_size, limit=2)
                started = time.perf_counter()
                measured = train_epoch(model, optimizer, fold['data']['train'], condition, args.batch_size)
                evaluate(model, fold['data']['validation'], condition, args.batch_size)
                elapsed = time.perf_counter() - started
                calibration[condition.name] = {'epoch_and_validation_seconds': elapsed,
                                               'train_seconds': measured['seconds']}
                print(f'Calibration {condition.name}: {elapsed:.2f}s per epoch + validation', flush=True)
                del model, optimizer
                torch.cuda.empty_cache()
            epoch_sum = sum(calibration[j['condition']]['epoch_and_validation_seconds'] for j in jobs)
            # Reserve evaluation/checkpoint and variable-fold overhead, without truncating jobs.
            evaluation_reserve = sum(calibration[j['condition']]['epoch_and_validation_seconds'] * 1.8 for j in jobs)
            epochs = args.epochs or max(1, int((args.hours * 3600 - evaluation_reserve) / (epoch_sum * 1.12)))
            plan = {'target_hours': args.hours, 'epochs': epochs, 'batch_size': args.batch_size,
                    'jobs': jobs, 'calibration': calibration, 'evaluation_reserve_seconds': evaluation_reserve,
                    'estimated_hours': (epochs * epoch_sum * 1.12 + evaluation_reserve) / 3600,
                    'budget_policy': 'one common fixed epoch count; finish every scheduled job; no hard wall-time cutoff',
                    'primary_jobs': 90, 'supplementary_jobs': 28,
                    'ideal_loso_seeds': '13 for all 17; 29,47 additionally for 2,5,15',
                    'smoke_excluded_from_metrics': True}
            write_json(plan_path, plan)
        print(f'LOCKED SCHEDULE: {len(jobs)} jobs x {epochs} epochs; estimate {plan["estimated_hours"]:.2f} hours', flush=True)
        last_subject = 2
        started = time.perf_counter()
        for index, job in enumerate(jobs):
            subject = job['subject']
            if subject != last_subject:
                del fold
                torch.cuda.empty_cache()
                fold = corpus.fold(subject)
                last_subject = subject
            fold_dir = output / 'folds' / f'subject_{subject:02d}'
            fold_dir.mkdir(parents=True, exist_ok=True)
            if not (fold_dir / 'split.npz').exists():
                np.savez_compressed(fold_dir / 'split.npz', **fold['indices'],
                                    mean=fold['mean'].numpy(), std=fold['std'].numpy())
                write_json(fold_dir / 'manifest.json', fold['summary'])
            condition = next(c for c in CONDITIONS if c.name == job['condition'])
            write_json(status_path, {'state': 'training', 'pid': os.getpid(), 'current_job': index + 1,
                'total_jobs': len(jobs), 'job': job, 'epochs': epochs, 'updated': datetime.now().isoformat()})
            train_job(fold, condition, subject, job['seed'], epochs, args.batch_size,
                      output / 'jobs' / condition.name / f'subject_{subject:02d}' / f'seed_{job["seed"]}')
            summarize(output)
        write_json(status_path, {'state': 'complete', 'pid': os.getpid(), 'completed_jobs': len(jobs),
            'training_seconds_this_invocation': time.perf_counter() - started, 'finished': datetime.now().isoformat()})
        print(f'COMPLETE: {output}', flush=True)
    except BaseException as error:
        write_json(status_path, {'state': 'failed', 'pid': os.getpid(), 'error': repr(error),
                                 'traceback': traceback.format_exc(), 'updated': datetime.now().isoformat()})
        raise
    finally:
        stop.set()
        monitor.join(timeout=12)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode', choices=['smoke', 'full'], default='full')
    parser.add_argument('--hours', type=float, default=5.5)
    parser.add_argument('--epochs', type=int, help='Explicit common epoch count instead of time calibration')
    parser.add_argument('--batch-size', type=int, default=256)
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--output', help='Existing run directory resumes its saved schedule/checkpoints')
    parser.add_argument('--rebuild-data', action='store_true')
    args = parser.parse_args()
    if args.hours <= 0 or args.batch_size < 2 or (args.epochs is not None and args.epochs < 1):
        parser.error('hours/epochs must be positive and batch size at least two')
    run(args)


if __name__ == '__main__':
    main()
