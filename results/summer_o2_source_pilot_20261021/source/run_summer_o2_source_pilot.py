"""Resumable, source-validation-only full-epoch pilot for seven summer O2 arms."""
import argparse
from datetime import datetime
import hashlib
import json
import msvcrt
from pathlib import Path
import shutil
import time

import torch

from o2_features import O2Corpus
from summer_o2_experiments import VARIANTS, run_job
from workstation_experiments import write_json


ROOT = Path(__file__).resolve().parent
SOURCES = ['run_summer_o2_source_pilot.py', 'summer_o2_experiments.py',
           'o2_features.py', 'models/summer_dynamic.py', 'models/mamba_whar.py',
           'summer_o1_experiments.py', 'o2_experiments.py', 'workstation_experiments.py']


def now():
    return datetime.now().astimezone().isoformat()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True)
    parser.add_argument('--epochs', type=int, default=8)
    parser.add_argument('--subjects', type=int, nargs='+', default=[2, 5])
    parser.add_argument('--seed', type=int, default=13)
    parser.add_argument('--batch-size', type=int, default=64)
    args = parser.parse_args()
    if args.epochs < 5 or args.epochs > 8 or not args.subjects:
        raise ValueError('Use 5-8 full epochs and at least one source split')
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    lock = (output / '.run.lock').open('a+b')
    if lock.tell() == 0:
        lock.write(b'0')
        lock.flush()
    lock.seek(0)
    msvcrt.locking(lock.fileno(), msvcrt.LK_NBLCK, 1)
    try:
        signature = {'sources': {name: hashlib.sha256((ROOT/name).read_bytes()).hexdigest()
                                 for name in SOURCES},
                     'epochs': args.epochs, 'subjects': args.subjects, 'seed': args.seed,
                     'batch_size': args.batch_size,
                     'conditions': [variant.name for variant in VARIANTS]}
        signature_file = output / 'signature.json'
        if signature_file.exists() and json.loads(signature_file.read_text()) != signature:
            raise RuntimeError('Pilot source or schedule changed; use a new output directory')
        write_json(signature_file, signature)
        source_dir = output / 'source'
        if not source_dir.exists():
            for name in SOURCES:
                dest = source_dir / name
                dest.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(ROOT/name, dest)
        if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
            raise RuntimeError('CUDA BF16 GPU required for matched summer pilots')
        torch.set_num_threads(8)
        torch.set_num_interop_threads(2)
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        corpus = O2Corpus()
        jobs = [(subject, variant) for subject in args.subjects for variant in VARIANTS]
        rows = []
        for number, (subject, variant) in enumerate(jobs, 1):
            destination = output / 'jobs' / variant.name / f'subject_{subject:02d}' / f'seed_{args.seed}'
            write_json(output/'status.json', {'state': 'running', 'updated': now(),
                'current_job': number, 'total_jobs': len(jobs), 'subject': subject,
                'condition': variant.name})
            fold = corpus.fold(subject, variant.kind)
            torch.cuda.reset_peak_memory_stats()
            start = time.perf_counter()
            result = run_job(fold, variant, subject, args.seed, args.epochs,
                             args.batch_size, destination, validation_only=True)
            seconds = time.perf_counter()-start
            row = {'subject': subject, 'condition': variant.name,
                   'validation_subject': fold['summary']['validation_subject'],
                   'best_epoch': result['best_epoch'],
                   'best_source_ideal_f1': result['best_validation_f1'],
                   'final_source_ideal_f1': result['history'][-1]['validation_ideal_f1'],
                   'source_self_f1_at_best_checkpoint': result['metrics']['val_self']['macro_f1'],
                   'seconds': seconds,
                   'peak_allocated_bytes': torch.cuda.max_memory_allocated()}
            if variant.graph == 'residual':
                state = torch.load(destination/'best.pt', map_location='cpu', weights_only=False)['model']
                row['query_weight_norm'] = float(state['q.weight'].float().norm())
                row['key_weight_norm'] = float(state['k.weight'].float().norm())
            rows.append(row)
            write_json(output/'summary.json', {'source_only': True, 'rows': rows})
            del fold
            torch.cuda.empty_cache()
        write_json(output/'completed.json', {'completed_jobs': len(jobs),
                   'expected_jobs': len(jobs), 'finished': now(), 'source_only': True})
        write_json(output/'status.json', {'state': 'complete', 'updated': now(),
                   'completed_jobs': len(jobs), 'total_jobs': len(jobs)})
        print(json.dumps({'output': str(output), 'completed_jobs': len(jobs),
                          'source_only': True}), flush=True)
    finally:
        lock.close()


if __name__ == '__main__':
    main()
