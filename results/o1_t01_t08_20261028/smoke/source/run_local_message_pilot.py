"""Foreground, CPU-staged six-job O1 pilot for a 4 GB CUDA GPU.

Source-only ideal validation selects checkpoints; no held-out test evaluation.
Resume with exactly the same command/output. Each epoch commits latest.pt.
"""
import argparse
from contextlib import nullcontext
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import random
import shutil
import sys
import time

import numpy as np
import torch
from torch.nn import functional as F
from models.summer_message_correction import ARMS, SummerMessageCorrection
from o2_features import O2Corpus, features
from summer_o1_experiments import scheduler_for
from workstation_experiments import metric_values, save_checkpoint, seed_all, write_json

ROOT = Path(__file__).resolve().parent
SOURCES = ('run_local_message_pilot.py', 'check_local_message_training.py',
           'models/summer_message_correction.py', 'models/summer_dynamic.py',
           'models/mamba_whar.py', 'o2_features.py', 'o2_experiments.py',
           'summer_o1_experiments.py', 'workstation_experiments.py',
           'prepare_workstation_data.py', 'requirements_local_gpu.txt')


def rng_state():
    return dict(cpu_rng=torch.get_rng_state(), cuda_rng=torch.cuda.get_rng_state_all(),
                python_rng=random.getstate(), numpy_rng=np.random.get_state())


def restore_rng(state):
    torch.set_rng_state(state['cpu_rng'])
    torch.cuda.set_rng_state_all(state['cuda_rng'])
    random.setstate(state['python_rng'])
    np.random.set_state(state['numpy_rng'])


def precision_context():
    return (torch.autocast('cuda', dtype=torch.bfloat16)
            if torch.cuda.is_bf16_supported() else nullcontext())


def build_fold(corpus, subject):
    eligible = [s for s in corpus.eligible_validation if s != subject]
    validation = next((s for s in eligible if s > subject), eligible[0])
    assert validation == {2: 3, 5: 8}[subject]
    ideal = corpus.scenarios == 'ideal'
    masks = {'train': ideal & corpus.valid & ~np.isin(corpus.subjects, [subject, validation]),
             'val_ideal': ideal & corpus.valid & (corpus.subjects == validation),
             'val_self': (corpus.scenarios == 'self') & corpus.valid & (corpus.subjects == validation)}
    indices = {k: np.flatnonzero(v) for k, v in masks.items()}
    data = {}
    for key, ix in indices.items():
        if not len(ix):
            raise ValueError(f'Empty {key}')
        # Feature extraction and storage stay in CPU RAM. Only a batch goes to CUDA.
        encoded = torch.empty((len(ix), 50, 9, 5), dtype=torch.float32)
        for start in range(0, len(ix), 256):
            encoded[start:start+256] = features(corpus.x[ix[start:start+256]], 'mag_dot')
        data[key] = {'x': encoded, 'y': corpus.y[ix], 'a': corpus.availability[ix]}
    mean = data['train']['x'].mean((0, 1), keepdim=True)
    std = data['train']['x'].std((0, 1), keepdim=True).clamp_min(1e-6)
    for part in data.values():
        part['x'].sub_(mean).div_(std).mul_(part['a'][:, None, :, None])
        if not torch.isfinite(part['x']).all():
            raise ValueError('Nonfinite normalized features')
    train_subjects = sorted(set(corpus.subjects[indices['train']].tolist()))
    assert subject not in train_subjects and validation not in train_subjects
    summary = {'test_subject_excluded': subject, 'validation_subject': validation,
               'train_subjects': train_subjects, 'counts': {k: len(v) for k, v in indices.items()},
               'recordings': {k: sorted(set(corpus.sources[v].tolist())) for k, v in indices.items()},
               'normalization': 'CPU float32 train-only mean/std, correction=1',
               'representation': 'mag_dot', 'test_data_evaluated': False}
    return {'data': data, 'indices': indices, 'mean': mean, 'std': std, 'summary': summary}


def batches(part, batch_size, shuffle=False):
    count = len(part['y'])
    order = torch.randperm(count) if shuffle else torch.arange(count)
    for start in range(0, count, batch_size):
        ix = order[start:start+batch_size]
        yield {k: v[ix].to('cuda') for k, v in part.items()}


@torch.inference_mode()
def evaluate(model, part, batch_size):
    model.eval()
    labels, predictions = [], []
    for batch in batches(part, batch_size):
        with precision_context():
            logits, _ = model(batch['x'], batch['a'])
        labels.append(batch['y'].cpu().numpy())
        predictions.append(logits.argmax(-1).cpu().numpy())
    y, pred = np.concatenate(labels), np.concatenate(predictions)
    return metric_values(y, pred), y, pred


def train_epoch(model, optimizer, part, batch_size):
    model.train()
    total = count = 0
    started = time.perf_counter()
    for batch in batches(part, batch_size, True):
        optimizer.zero_grad(set_to_none=True)
        with precision_context():
            logits, _ = model(batch['x'], batch['a'])
            loss = F.cross_entropy(logits, batch['y'])
        if not torch.isfinite(loss):
            raise FloatingPointError('Nonfinite training loss')
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
        optimizer.step()
        total += float(loss.detach()) * len(batch['y'])
        count += len(batch['y'])
    torch.cuda.synchronize()
    return {'loss': total/count, 'windows': count, 'seconds': time.perf_counter()-started}


def run_job(fold, arm, subject, args, output):
    output.mkdir(parents=True, exist_ok=True)
    config = {'arm': arm, 'subject': subject, 'seed': 13, 'epochs': args.epochs,
              'batch_size': args.batch_size, 'features': 'mag_dot', 'hidden': 16,
              'dropout': .5, 'optimizer': 'Adam', 'lr': 1e-4, 'weight_decay': 5e-4,
              'selection': 'source ideal fixed-33 macro-F1', 'data_device': 'cpu',
              'precision': 'bf16' if torch.cuda.is_bf16_supported() else 'fp32'}
    if (output/'result.json').exists():
        result = json.loads((output/'result.json').read_text())
        if result['config'] != config or not all((output/p).exists() for p in ('latest.pt', 'best.pt', 'epochs.jsonl')):
            raise ValueError('Completed job config/artifacts mismatch')
        return result
    seed_all(13)
    model = SummerMessageCorrection(arm).cuda()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=5e-4, fused=True)
    scheduler = scheduler_for(optimizer, args.epochs)
    history, first, best, best_epoch = [], 1, -1., 0
    if (output/'latest.pt').exists():
        state = torch.load(output/'latest.pt', map_location='cpu', weights_only=False)
        if state['config'] != config:
            raise ValueError('Resume configuration changed')
        model.load_state_dict(state['model'])
        optimizer.load_state_dict(state['optimizer'])
        scheduler.load_state_dict(state['scheduler'])
        restore_rng(state)
        history, first = state['history'], state['epoch']+1
        best, best_epoch = state['best'], state['best_epoch']
        print(f'Resuming {arm}, subject {subject}, epoch {first}', flush=True)
        del state
    write_json(output/'config.json', config)
    # Reconcile the text history to the last committed checkpoint after interruption.
    with (output/'epochs.jsonl').open('w', encoding='utf-8') as log:
        for row in history:
            log.write(json.dumps(row)+'\n')
    started = time.perf_counter()
    torch.cuda.reset_peak_memory_stats()
    for epoch in range(first, args.epochs+1):
        train = train_epoch(model, optimizer, fold['data']['train'], args.batch_size)
        val, _, _ = evaluate(model, fold['data']['val_ideal'], args.batch_size)
        if val['macro_f1'] > best:
            best, best_epoch = val['macro_f1'], epoch
            save_checkpoint(output/'best.pt', {'model': model.state_dict(), 'config': config,
                            'epoch': epoch, 'validation_f1': best, 'mean': fold['mean'], 'std': fold['std']})
        scheduler.step()
        row = {'epoch': epoch, 'train': train, 'validation_ideal_f1': val['macro_f1'],
               'lr': optimizer.param_groups[0]['lr'],
               'peak_cuda_allocated_mb': torch.cuda.max_memory_allocated()/1024**2}
        history.append(row)
        save_checkpoint(output/'latest.pt', {'model': model.state_dict(),
            'optimizer': optimizer.state_dict(), 'scheduler': scheduler.state_dict(),
            'config': config, 'epoch': epoch, 'best': best, 'best_epoch': best_epoch,
            'history': history, 'mean': fold['mean'], 'std': fold['std'], **rng_state()})
        with (output/'epochs.jsonl').open('a', encoding='utf-8') as log:
            log.write(json.dumps(row)+'\n')
        print(f'{arm} S{subject:02d} epoch {epoch}/{args.epochs} '
              f'loss={train["loss"]:.4f} source_val_F1={val["macro_f1"]:.4f} '
              f'train={train["seconds"]:.1f}s VRAM={row["peak_cuda_allocated_mb"]:.0f}MB [checkpoint saved]', flush=True)
    model.load_state_dict(torch.load(output/'best.pt', map_location='cpu', weights_only=False)['model'])
    metrics = {}
    for key in ('val_ideal', 'val_self'):
        metrics[key], y, pred = evaluate(model, fold['data'][key], args.batch_size)
        np.savez_compressed(output/f'predictions_{key}.npz', labels=y, predictions=pred,
                            corpus_indices=fold['indices'][key])
    with torch.inference_mode():
        batch = next(batches(fold['data']['val_ideal'], min(args.batch_size, 32)))
        _, _, d = model(batch['x'], batch['a'], return_diagnostics=True)
        diagnostics = {k+'_rms': float(d[k].float().square().mean().sqrt())
                       for k in ('local', 'message', 'correction')}
    result = {'config': config, 'best_validation_f1': best, 'best_epoch': best_epoch,
              'metrics': metrics, 'diagnostics': diagnostics, 'history': history,
              'split': fold['summary'], 'seconds_this_invocation': time.perf_counter()-started,
              'parameter_count': sum(p.numel() for p in model.parameters()),
              'note': 'Identical modules allocated for initialization; unused modules receive no gradients.'}
    write_json(output/'result.json', result)
    del model, optimizer, scheduler
    torch.cuda.empty_cache()
    return result


def summarize(output):
    rows = [json.loads(p.read_text()) for p in (output/'jobs').glob('*/*/result.json')]
    summary = {}
    for arm in ARMS:
        selected = [r for r in rows if r['config']['arm'] == arm]
        summary[arm] = {'completed_jobs': len(selected), 'expected_jobs': 2,
                       'source_ideal_f1': float(np.mean([r['metrics']['val_ideal']['macro_f1'] for r in selected])) if selected else None,
                       'source_self_f1': float(np.mean([r['metrics']['val_self']['macro_f1'] for r in selected])) if selected else None}
    write_json(output/'summary.json', summary)
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', required=True)
    parser.add_argument('--epochs', type=int, default=8)
    parser.add_argument('--batch-size', type=int, default=64)
    args = parser.parse_args()
    if args.epochs < 4 or args.batch_size < 2:
        raise ValueError('Require at least four epochs and batch size two')
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA is required; no silent CPU fallback')
    torch.set_num_threads(4)
    torch.set_num_interop_threads(2)
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = True
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    import msvcrt
    with (output/'.run.lock').open('a+b') as lock:
        if lock.tell() == 0:
            lock.write(b'0'); lock.flush()
        lock.seek(0)
        msvcrt.locking(lock.fileno(), msvcrt.LK_NBLCK, 1)
        def status(state, **extra):
            write_json(output/'status.json', {'state': state, 'pid': os.getpid(),
                'updated': datetime.now().astimezone().isoformat(), **extra})
        try:
            fingerprint = {'sources': {p: hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in SOURCES},
                'cache_audit': hashlib.sha256((ROOT/'dataset/workstation_o1/audit.json').read_bytes()).hexdigest(),
                'schedule': {'arms': list(ARMS), 'subjects': [2, 5], 'seed': 13,
                             'epochs': args.epochs, 'batch_size': args.batch_size}}
            preflight = json.loads((output/'preflight.json').read_text())
            if not preflight.get('passed') or preflight.get('source_hashes') != fingerprint['sources']:
                raise ValueError('A passing preflight for the current source is required')
            if (output/'fingerprint.json').exists() and json.loads((output/'fingerprint.json').read_text()) != fingerprint:
                raise ValueError('Source/data/schedule mismatch; use original code to resume')
            write_json(output/'fingerprint.json', fingerprint)
            for name in SOURCES:
                dest = output/'source'/name
                dest.parent.mkdir(parents=True, exist_ok=True)
                if not dest.exists():
                    shutil.copy2(ROOT/name, dest)
            write_json(output/'system.json', {'python': sys.version, 'executable': sys.executable,
                'torch': torch.__version__, 'cuda': torch.version.cuda, 'gpu': torch.cuda.get_device_name(),
                'bf16': torch.cuda.is_bf16_supported(), 'total_vram_mb': torch.cuda.get_device_properties(0).total_memory/1024**2})
            status('loading_cpu_data', completed_jobs=0, total_jobs=6)
            print('Loading REALDISP into CPU RAM; GPU receives one batch at a time.', flush=True)
            corpus = O2Corpus()
            count = 0
            for subject in (2, 5):
                print(f'Preparing source-only fold S{subject:02d}...', flush=True)
                fold = build_fold(corpus, subject)
                write_json(output/f'fold_{subject:02d}.json', fold['summary'])
                save_checkpoint(output/f'normalization_{subject:02d}.pt', {'mean': fold['mean'], 'std': fold['std'], 'indices': fold['indices']})
                for arm in ARMS:
                    status('training', current_job=count+1, completed_jobs=count, total_jobs=6, subject=subject, arm=arm)
                    run_job(fold, arm, subject, args, output/'jobs'/arm/f'subject_{subject:02d}')
                    count += 1
                    summarize(output)
                del fold
            summary = summarize(output)
            text = '# O1 latent message-correction pilot\n\nEight-epoch source-only pilot (unless overridden in fingerprint). No held-out placement evaluation.\n\n| Arm | Source ideal F1 | Source self F1 |\n|---|---:|---:|\n'
            for arm, row in summary.items():
                text += f'| {arm} | {row["source_ideal_f1"]*100:.2f}% | {row["source_self_f1"]*100:.2f}% |\n'
            text += '\nReview paired fold effects and correction diagnostics before selecting a full study. This pilot does not establish test superiority or a Mamba/SSM result. CPU staging changes numerical/data-order details from historical GPU-resident runs; use the fresh matched controls.\n'
            (output/'PILOT_REVIEW.md').write_text(text, encoding='utf-8')
            write_json(output/'completed.json', {'completed_jobs': count, 'expected_jobs': 6})
            status('complete', completed_jobs=count, total_jobs=6)
        except BaseException as error:
            status('interrupted' if isinstance(error, KeyboardInterrupt) else 'failed', error=repr(error))
            raise


if __name__ == '__main__':
    main()
