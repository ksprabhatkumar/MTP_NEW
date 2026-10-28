"""Single-entry Objective 2 runner: preflight, smoke, calibrate, sequential training."""
import argparse
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
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
    def write(self, value):
        for stream in self.streams:
            stream.write(value)
            stream.flush()
    def flush(self):
        for stream in self.streams:
            stream.flush()


def run(args):
    import msvcrt
    import numpy as np
    import torch
    from o2_features import O2Corpus, INVALID, features, random_rotations, rotate_raw
    from o2_experiments import (VARIANTS, Variant, model_for, optimizer_for,
                                train_epoch, evaluate, run_job, summarize)
    from workstation_experiments import seed_all, write_json
    from run_training import telemetry

    output = Path(args.output).resolve() if args.output else ROOT / 'results' / (
        datetime.now().strftime('%Y%m%d_%H%M%S') + '_o2')
    output.mkdir(parents=True, exist_ok=True)
    lock = (output / '.run.lock').open('a+b')
    if lock.tell() == 0:
        lock.write(b'0')
        lock.flush()
    lock.seek(0)
    try:
        msvcrt.locking(lock.fileno(), msvcrt.LK_NBLCK, 1)
    except OSError as error:
        raise RuntimeError(f'Run directory already active: {output}') from error
    run_log = (output / 'run.log').open('a', encoding='utf-8', buffering=1)
    sys.stdout, sys.stderr = Tee(sys.stdout, run_log), Tee(sys.stderr, run_log)
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError('Objective 2 requires CUDA BF16; no CPU fallback.')
    torch.set_num_threads(min(8, os.cpu_count() or 1))
    torch.set_num_interop_threads(2)
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    print('O2 OUTPUT', output, 'GPU', torch.cuda.get_device_name(), flush=True)
    with (output / 'requirements.lock.txt').open('w') as lock_file:
        subprocess.run([sys.executable, '-m', 'pip', 'freeze'], stdout=lock_file, check=True)
    source = output / 'source'
    paths = [ROOT / name for name in ('run_o2.py','o2_experiments.py','o2_features.py',
             'workstation_experiments.py','run_training.py','prepare_workstation_data.py',
             'correct_o1_quality_results.py','NEXT_STEP_PLAN.md')]
    paths += list((ROOT / 'models').glob('*.py')) + list((ROOT / 'tests').glob('*.py'))
    if not source.exists():
        for path in paths:
            target = source / path.relative_to(ROOT)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, target)
    hashes = {str(path.relative_to(source)): hashlib.sha256(path.read_bytes()).hexdigest()
              for path in source.rglob('*') if path.is_file()}
    for relative, expected in hashes.items():
        if hashlib.sha256((ROOT / relative).read_bytes()).hexdigest() != expected:
            raise ValueError(f'Source changed since O2 run started: {relative}')
    write_json(output / 'source_hashes.json', hashes)
    preflight = ROOT / 'results' / 'o2_preflight_20261017'
    invalid = json.loads((preflight / 'invalid_recordings.json').read_text())
    if {record['source'] for record in invalid} != INVALID or not all(record['all_zero'] for record in invalid):
        raise ValueError('Preflight zero-signal audit has changed')
    if not (preflight / 'corrected_o1_primary.csv').exists():
        raise ValueError('Corrected O1 comparison is missing')
    audit_source = ROOT / 'dataset' / 'workstation_o1' / 'audit.json'
    audit_digest = hashlib.sha256(audit_source.read_bytes()).hexdigest()
    audit_target = output / 'data_audit.json'
    if audit_target.exists() and hashlib.sha256(audit_target.read_bytes()).hexdigest() != audit_digest:
        raise ValueError('Data audit changed; use a fresh run folder')
    shutil.copy2(audit_source, audit_target)
    write_json(output / 'excluded_recordings.json', invalid)
    stop = threading.Event()
    monitor = threading.Thread(target=telemetry, args=(output, stop), daemon=True)
    monitor.start()
    status = output / 'status.json'
    try:
        write_json(status, {'state':'preflight','pid':os.getpid(),'started':datetime.now().isoformat()})
        tests = subprocess.run([sys.executable, '-m', 'unittest', 'discover', '-s', 'tests', '-v'],
            capture_output=True, text=True)
        (output / 'unit_tests.log').write_text(tests.stdout + tests.stderr)
        if tests.returncode:
            raise RuntimeError('Unit tests failed; see unit_tests.log')
        corpus = O2Corpus()
        write_json(output / 'filtered_data_manifest.json', corpus.valid_manifest)
        print('Retained', corpus.valid_manifest['retained_windows'], 'windows; validation subjects',
              corpus.eligible_validation, flush=True)
        if not (output / 'smoke' / 'passed.json').exists():
            for variant in VARIANTS:
                fold = corpus.fold(2, variant.kind)
                assert fold['summary']['validation_subject'] == 3
                assert not set(fold['summary']['recordings']['train']) & set(fold['summary']['recordings']['ideal'])
                run_job(fold, variant, 2, 13, 2, args.batch_size,
                        output / 'smoke' / variant.name, limit=2)
                del fold
                torch.cuda.empty_cache()
            write_json(output / 'smoke' / 'passed.json',
                {'passed': True, 'variants': [v.name for v in VARIANTS],
                 'epochs': 2, 'batch_limit': 2, 'research_result': False})
        if args.mode == 'smoke':
            write_json(status, {'state':'smoke_complete','pid':os.getpid(),
                                'finished':datetime.now().isoformat()})
            return
        jobs = []
        for subject in [2,5,15] + [s for s in range(1,18) if s not in (2,5,15)]:
            variants = VARIANTS if subject in (2,5,15) else [v for v in VARIANTS if v.kind in ('raw','mag_dot')]
            seeds = (13,29,47) if subject in (2,5,15) else (13,)
            jobs += [{'subject':subject,'kind':v.kind,'graph':v.graph,'seed':seed}
                     for v in variants for seed in seeds]
        if len(jobs) != 128:
            raise AssertionError(len(jobs))
        schedule_path = output / 'schedule.json'
        if schedule_path.exists():
            schedule = json.loads(schedule_path.read_text())
            if schedule['batch_size'] != args.batch_size:
                raise ValueError('Resume batch size mismatch')
            epochs, jobs = schedule['epochs'], schedule['jobs']
        else:
            calibration = {}
            print('Calibrating full training and source validation, no test scores.', flush=True)
            for variant in VARIANTS:
                fold = corpus.fold(2, variant.kind)
                seed_all(101)
                model = model_for(variant)
                optimizer = optimizer_for(model)
                train_epoch(model, optimizer, fold, variant, args.batch_size, limit=2)
                started = time.perf_counter()
                train_epoch(model, optimizer, fold, variant, args.batch_size)
                evaluate(model, fold, 'val_ideal', args.batch_size)
                evaluate(model, fold, 'val_self', args.batch_size)
                calibration[variant.name] = time.perf_counter() - started
                print('Calibration', variant.name, calibration[variant.name], 'sec', flush=True)
                del model, optimizer, fold
                torch.cuda.empty_cache()
            epoch_sum = sum(calibration[f"{job['kind']}_{job['graph']}"] for job in jobs)
            reserve = sum(calibration[f"{job['kind']}_{job['graph']}"] * 2.2 for job in jobs)
            epochs = args.epochs or max(20, min(53, int((args.hours * 3600 - reserve) / (epoch_sum * 1.12))))
            schedule = {'jobs':jobs,'epochs':epochs,'batch_size':args.batch_size,
                'calibration':calibration,'target_hours':args.hours,
                'estimated_hours':(epoch_sum * epochs * 1.12 + reserve) / 3600,
                'selection':'source-validation ideal/self shared-class F1 with ideal floor',
                'status':'exploratory: induced subjects inspected during O1 design',
                'primary_jobs':72,'supplementary_jobs':56}
            write_json(schedule_path, schedule)
        print('LOCKED O2 SCHEDULE',len(jobs),'jobs x',epochs,'epochs; estimate',
              schedule['estimated_hours'],'hours',flush=True)
        started = time.perf_counter()
        for number, job in enumerate(jobs, 1):
            variant = Variant(job['kind'], job['graph'])
            fold = corpus.fold(job['subject'], variant.kind)
            fold_path = output / 'folds' / f"subject_{job['subject']:02d}" / variant.kind
            fold_path.mkdir(parents=True, exist_ok=True)
            if not (fold_path / 'split.npz').exists():
                np.savez_compressed(fold_path / 'split.npz', **fold['indices'],
                    mean=fold['mean'].numpy(), std=fold['std'].numpy())
                write_json(fold_path / 'manifest.json', fold['summary'])
            write_json(status, {'state':'training','pid':os.getpid(),'current_job':number,
                'total_jobs':len(jobs),'job':job,'epochs':epochs,'updated':datetime.now().isoformat()})
            run_job(fold, variant, job['subject'], job['seed'], epochs, args.batch_size,
                    output / 'jobs' / variant.name / f"subject_{job['subject']:02d}" / f"seed_{job['seed']}")
            summarize(output)
            del fold
            torch.cuda.empty_cache()
        plot_python = args.plot_python or shutil.which('python') or sys.executable
        with (output / 'plotting.log').open('w') as plot_log:
            plot = subprocess.run([plot_python, str(ROOT / 'plot_o2_results.py'), str(output)],
                stdout=plot_log, stderr=subprocess.STDOUT)
        if plot.returncode:
            print('Optional plots failed; see plotting.log', flush=True)
        write_json(status, {'state':'complete','pid':os.getpid(),'completed_jobs':len(jobs),
            'seconds_this_invocation':time.perf_counter()-started,'finished':datetime.now().isoformat()})
    except BaseException as error:
        write_json(status, {'state':'failed','pid':os.getpid(),'error':repr(error),
            'traceback':traceback.format_exc(),'updated':datetime.now().isoformat()})
        raise
    finally:
        stop.set()
        monitor.join(timeout=12)
        lock.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode', choices=('smoke','full'), default='full')
    parser.add_argument('--hours', type=float, default=5.5)
    parser.add_argument('--epochs', type=int)
    parser.add_argument('--batch-size', type=int, default=256)
    parser.add_argument('--output')
    parser.add_argument('--plot-python')
    args = parser.parse_args()
    if args.batch_size < 2 or args.hours <= 0 or args.epochs is not None and args.epochs < 1:
        parser.error('invalid compute budget')
    run(args)
