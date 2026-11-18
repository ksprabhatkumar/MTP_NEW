"""Sequential, resumable summer-based Objective 2 runner.

Prepare while O1 trains; run GPU modes only after the O1 session is complete.
The old run_o2.py uses a different spatial pipeline and is not reused here.
"""

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
O1 = ROOT / 'results' / 'summer_o1_session_20261017_232945'
SOURCES = ['run_summer_o2.py', 'summer_o2_experiments.py',
           'check_summer_o2_invariance.py', 'verify_summer_o2_bridge.py',
           'analyze_summer_o2.py', 'finalize_summer_o3_config.py',
           'models/summer_dynamic.py', 'models/mamba_whar.py',
           'summer_o1_experiments.py', 'o2_features.py', 'o2_experiments.py',
           'workstation_experiments.py', 'prepare_workstation_data.py',
           'run_training.py']


def o1_complete():
    outer = O1 / 'status.json'
    inner = O1 / 'training' / 'completed.json'
    if not outer.exists() or not inner.exists():
        return False
    return (json.loads(outer.read_text())['state'] == 'complete'
            and json.loads(inner.read_text())['completed_jobs'] == 126
            and (O1 / 'training' / 'ANALYSIS.md').exists())


def publish(output, data):
    from workstation_experiments import write_json
    state = dict(data, pid=os.getpid(), updated=datetime.now().astimezone().isoformat())
    write_json(output / 'status.json', state)
    handoff = ('SUMMER-BASED OBJECTIVE 2 SESSION\n'
        + json.dumps(state, indent=2) + '\n\n'
        + f'Output: {output}\nO1 dependency: {O1}\n'
        + 'Source-only validation chooses checkpoints; held-out placements are evaluation only.\n'
        + 'Seven feature/graph conditions, subjects 2/5/15, seeds 13/29/47.\n'
        + 'Resume the same output directory only after confirming no active writer.\n')
    (output / 'TRAINING_HANDOFF.txt').write_text(handoff, encoding='utf-8')


def run(args):
    import msvcrt
    import numpy as np
    import torch
    from o2_features import O2Corpus
    from summer_o2_experiments import VARIANTS, BRIDGE, model_for, run_job, summarize
    from workstation_experiments import seed_all, write_json
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
    monitor_stop = threading.Event()
    monitor = None
    try:
        if not o1_complete():
            raise RuntimeError('O1 must finish all 126 jobs and analysis before O2 GPU work')
        if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
            raise RuntimeError('CUDA BF16 GPU required')
        torch.set_num_threads(8)
        torch.set_num_interop_threads(2)
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        print('SUMMER O2', args.mode, output, torch.cuda.get_device_name(), flush=True)
        hashes = {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
                  for name in SOURCES}
        fingerprint = {'sources': hashes,
            'cache_audit': hashlib.sha256(
                (ROOT / 'dataset/workstation_o1/audit.json').read_bytes()).hexdigest(),
            'o1_run': str(O1), 'o1_schedule': hashlib.sha256(
                (O1 / 'training' / 'schedule.json').read_bytes()).hexdigest()}
        if (output / 'fingerprint.json').exists():
            if json.loads((output / 'fingerprint.json').read_text()) != fingerprint:
                raise ValueError('O2 source, data or O1 schedule changed')
        write_json(output / 'fingerprint.json', fingerprint)
        if not (output / 'source').exists():
            for name in SOURCES:
                destination = output / 'source' / name
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(ROOT / name, destination)
            shutil.copy2(ROOT / 'NEXT_TWO_OBJECTIVES_TRAINING_PLAN.md',
                         output / 'plan_at_start.md')
        write_json(output / 'system.json', {'python': sys.version,
            'executable': sys.executable, 'torch': torch.__version__,
            'cuda': torch.version.cuda, 'gpu': torch.cuda.get_device_name(),
            'batch_size': args.batch_size})
        with (output / 'requirements.lock.txt').open('w') as req:
            subprocess.run([sys.executable, '-m', 'pip', 'freeze'],
                           stdout=req, check=True)
        monitor = threading.Thread(target=telemetry,
                                   args=(output, monitor_stop), daemon=True)
        monitor.start()
        publish(output, {'state': 'preflight', 'plan_step': 'P5.1'})
        corpus = O2Corpus()
        folds = {}
        for subject in (2, 5, 15):
            fold = corpus.fold(subject, 'raw')
            summary = fold['summary']
            if subject in summary['train_subjects'] or summary['validation_subject'] in summary['train_subjects']:
                raise ValueError('O2 split leaked validation/test subject')
            folds[str(subject)] = summary
            del fold
            torch.cuda.empty_cache()
        write_json(output / 'fold_preflight.json', folds)
        if args.mode == 'preflight':
            write_json(output / 'passed.json', {'passed': True,
                'subjects': [2, 5, 15], 'variants': [v.name for v in VARIANTS]})
        elif args.mode in ('smoke', 'bridge'):
            variants = VARIANTS if args.mode == 'smoke' else [BRIDGE]
            for variant in variants:
                fold = corpus.fold(2, variant.kind)
                fold['sources'] = {key: corpus.sources[ix]
                                   for key, ix in fold['indices'].items()}
                epochs = 2 if args.mode == 'smoke' else 31
                limit = 2 if args.mode == 'smoke' else None
                run_job(fold, variant, 2, 13, epochs, args.batch_size,
                        output / variant.name, limit=limit,
                        validation_only=args.mode == 'bridge')
                del fold
                torch.cuda.empty_cache()
            write_json(output / 'passed.json', {'passed': True, 'mode': args.mode,
                'variants': [v.name for v in variants]})
        else:
            bridge = Path(args.bridge).resolve() / BRIDGE.name / 'result.json'
            if not bridge.is_file():
                raise RuntimeError('Run the O2 raw/physical bridge before full training')
            # The bridge is a readiness check. Its curve and final weights must
            # be reviewed against the matching O1 job before this mode starts.
            review_file = Path(args.bridge).resolve() / 'bridge_review.json'
            if not review_file.is_file() or not json.loads(review_file.read_text()).get('passed'):
                raise RuntimeError('The O2 raw/physical bridge parity review has not passed')
            o1_schedule = json.loads((O1 / 'training' / 'schedule.json').read_text())
            epochs = o1_schedule['epochs']
            if epochs != 31 or args.batch_size != o1_schedule['batch_size']:
                raise ValueError('O1/O2 epoch or batch budget mismatch')
            jobs = [{'condition': variant.name, 'subject': subject, 'seed': seed}
                for subject in (2, 5, 15) for variant in VARIANTS
                for seed in (13, 29, 47)]
            schedule_file = output / 'schedule.json'
            schedule = {'jobs': jobs, 'epochs': epochs,
                'batch_size': args.batch_size, 'o1_run': str(O1),
                'selection': 'source ideal validation 33-class macro-F1',
                'bridge_result': str(bridge)}
            if schedule_file.exists():
                if json.loads(schedule_file.read_text()) != schedule:
                    raise ValueError('Summer O2 schedule changed')
            else:
                write_json(schedule_file, schedule)
            for number, job in enumerate(jobs, 1):
                destination = output / 'jobs' / job['condition'] / f'subject_{job["subject"]:02d}' / f'seed_{job["seed"]}'
                if (destination / 'result.json').exists():
                    continue
                variant = next(v for v in VARIANTS if v.name == job['condition'])
                fold = corpus.fold(job['subject'], variant.kind)
                fold['sources'] = {key: corpus.sources[ix]
                                   for key, ix in fold['indices'].items()}
                publish(output, {'state': 'training', 'plan_step': 'P5.5',
                    'current_job': number, 'total_jobs': len(jobs), 'job': job,
                    'epochs': epochs})
                run_job(fold, variant, job['subject'], job['seed'], epochs,
                        args.batch_size, destination)
                summarize(output)
                del fold
                torch.cuda.empty_cache()
            write_json(output / 'completed.json', {'completed_jobs': len(jobs),
                'finished': datetime.now().astimezone().isoformat()})
        publish(output, {'state': 'complete', 'mode': args.mode,
                         'plan_step': 'P5.5' if args.mode == 'full' else 'P5.3'})
    except BaseException as error:
        publish(output, {'state': 'failed', 'mode': args.mode,
            'error': repr(error), 'traceback': traceback.format_exc()})
        raise
    finally:
        monitor_stop.set()
        if monitor:
            monitor.join(timeout=12)
        lock.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode', choices=('plan', 'preflight', 'smoke', 'bridge', 'full'),
                        required=True)
    parser.add_argument('--output')
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--bridge')
    args = parser.parse_args()
    if args.mode == 'plan':
        from summer_o2_experiments import VARIANTS
        print(json.dumps({'o1_complete': o1_complete(), 'jobs': len(VARIANTS) * 9,
            'variants': [v.name for v in VARIANTS], 'gpu_used': False}, indent=2))
    else:
        if not args.output:
            parser.error('--output is required for GPU modes')
        if args.mode == 'full' and not args.bridge:
            parser.error('--bridge is required for full mode')
        run(args)
