"""Isolated source-only test of query/key Adam decay in the summer O2 router."""
import argparse
from datetime import datetime
import hashlib
import json
import msvcrt
from pathlib import Path
import random
import shutil
import time

import numpy as np
import torch
from torch.nn import functional as F

from audit_summer_routing import effect, mix
from o2_features import O2Corpus
from summer_o1_experiments import scheduler_for
from summer_o2_experiments import Variant, model_for, train_epoch
from o2_experiments import evaluate
from workstation_experiments import save_checkpoint, seed_all, write_json


ROOT = Path(__file__).resolve().parent
SOURCES = ['pilot_affinity_decay.py', 'audit_summer_routing.py',
           'summer_o2_experiments.py', 'summer_o1_experiments.py',
           'o2_features.py', 'models/summer_dynamic.py', 'models/mamba_whar.py',
           'o2_experiments.py', 'workstation_experiments.py']
ARMS = [('residual_standard', 'residual', False),
        ('residual_qk_nodecay', 'residual', True),
        ('physical_standard', 'physical', False),
        ('none_standard', 'none', False)]


def optimizer_for(model, qk_nodecay):
    if not qk_nodecay:
        return torch.optim.Adam(model.parameters(), lr=1e-4,
                                weight_decay=5e-4, fused=True)
    special = [model.q.weight, model.k.weight]
    special_ids = {id(p) for p in special}
    ordinary = [p for p in model.parameters() if id(p) not in special_ids]
    if len(ordinary) + len(special) != len(list(model.parameters())):
        raise RuntimeError('Optimizer parameter groups overlap or omit a parameter')
    return torch.optim.Adam([
        {'params': ordinary, 'weight_decay': 5e-4},
        {'params': special, 'weight_decay': 0.0}], lr=1e-4, fused=True)


def probe(model, fold):
    part = fold['data']['val_ideal']
    eligible = torch.where(part['a'].all(-1))[0]
    ix = eligible[torch.linspace(0, len(eligible)-1, min(96, len(eligible)),
                                 device='cuda').long()]
    x, a, y = part['x'][ix], part['a'][ix], part['y'][ix]
    with torch.inference_mode():
        logits, _, diagnostics = model(x, a, return_diagnostics=True)
        adjacency = diagnostics['adjacency'].float()
        dynamic = diagnostics['dynamic'].float()
        changed, _ = mix(model, dynamic.roll(1, 0), a)
        donor = effect(model, x, a, y, adjacency, logits, changed)
        off = 1-torch.eye(9, device=x.device)
        forced = (off/8)[None, None].expand_as(adjacency)
        forced_effect = effect(model, x, a, y, adjacency, logits, forced)
        return {'query_weight_norm': float(model.q.weight.float().norm()),
                'key_weight_norm': float(model.k.weight.float().norm()),
                'between_window_adjacency_std': float(adjacency.std(0).mean()),
                'effective_branch_mass': diagnostics['effective_branch_mass'].cpu().tolist(),
                'donor_affinity': donor,
                'forced_uniform_final_adjacency': forced_effect,
                'sample_windows': len(ix)}


def run_one(fold, arm, subject, seed, epochs, batch_size, output):
    name, graph, qk_nodecay = arm
    destination = output/'jobs'/name/f'subject_{subject:02d}'/f'seed_{seed}'
    destination.mkdir(parents=True, exist_ok=True)
    result_file = destination/'result.json'
    if result_file.exists():
        return json.loads(result_file.read_text(encoding='utf-8'))
    seed_all(seed)
    variant = Variant('mag_dot', graph)
    model = model_for(variant)
    optimizer = optimizer_for(model, qk_nodecay)
    scheduler = scheduler_for(optimizer, epochs)
    config = {'arm': name, 'graph': graph, 'feature': 'mag_dot',
              'qk_weight_decay': 0.0 if qk_nodecay else 5e-4,
              'other_weight_decay': 5e-4, 'optimizer': 'Adam fused',
              'lr': 1e-4, 'scheduler': 'OneCycleLR once/epoch',
              'epochs': epochs, 'batch_size': batch_size, 'seed': seed,
              'held_out_subject': subject, 'validation_only': True,
              'selection': 'distinct source subject ideal 33-class macro-F1'}
    write_json(destination/'config.json', config)
    first, best, best_epoch, history = 1, -1.0, 0, []
    latest = destination/'latest.pt'
    if latest.exists():
        state = torch.load(latest, map_location='cpu', weights_only=False)
        if state['config'] != config:
            raise RuntimeError('Checkpoint config changed; use a new output directory')
        model.load_state_dict(state['model'])
        optimizer.load_state_dict(state['optimizer'])
        scheduler.load_state_dict(state['scheduler'])
        first, best, best_epoch, history = state['epoch']+1, state['best'], state['best_epoch'], state['history']
        torch.set_rng_state(state['cpu_rng'])
        torch.cuda.set_rng_state_all(state['cuda_rng'])
        random.setstate(state['python_rng'])
        np.random.set_state(state['numpy_rng'])
    started = time.perf_counter()
    with (destination/'epochs.jsonl').open('a', encoding='utf-8', buffering=1) as log:
        for epoch in range(first, epochs+1):
            train = train_epoch(model, optimizer, fold, variant, batch_size)
            validation, _, _ = evaluate(model, fold, 'val_ideal', batch_size)
            row = {'epoch': epoch, 'train': train,
                   'validation_ideal_f1': validation['macro_f1'],
                   'lr': optimizer.param_groups[0]['lr'],
                   'query_weight_norm': float(model.q.weight.float().norm()),
                   'key_weight_norm': float(model.k.weight.float().norm())}
            history.append(row)
            log.write(json.dumps(row)+'\n')
            if validation['macro_f1'] > best:
                best, best_epoch = validation['macro_f1'], epoch
                save_checkpoint(destination/'best.pt',
                    {'model': model.state_dict(), 'config': config, 'epoch': epoch,
                     'validation_f1': best, 'mean': fold['mean'], 'std': fold['std']})
            scheduler.step()
            save_checkpoint(latest, {'model': model.state_dict(), 'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict(), 'config': config,
                'epoch': epoch, 'best': best, 'best_epoch': best_epoch, 'history': history,
                'cpu_rng': torch.get_rng_state(), 'cuda_rng': torch.cuda.get_rng_state_all(),
                'python_rng': random.getstate(), 'numpy_rng': np.random.get_state()})
            print(f'{name} S{subject:02d} epoch={epoch}/{epochs} '
                  f'source_F1={validation["macro_f1"]:.4f} q={row["query_weight_norm"]:.4f}', flush=True)
    last_probe = probe(model.eval(), fold)
    model.load_state_dict(torch.load(destination/'best.pt', map_location='cpu', weights_only=False)['model'])
    model.eval()
    source_ideal, _, _ = evaluate(model, fold, 'val_ideal', batch_size)
    source_self, _, _ = evaluate(model, fold, 'val_self', batch_size)
    result = {'arm': name, 'subject': subject, 'seed': seed,
              'validation_subject': fold['summary']['validation_subject'],
              'best_epoch': best_epoch, 'best_source_ideal_f1': best,
              'source_ideal_f1_at_best': source_ideal['macro_f1'],
              'source_self_f1_at_best': source_self['macro_f1'],
              'best_probe': probe(model, fold), 'last_probe': last_probe,
              'history': history, 'config': config,
              'seconds': time.perf_counter()-started}
    write_json(result_file, result)
    del model, optimizer
    torch.cuda.empty_cache()
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True)
    parser.add_argument('--epochs', type=int, default=8)
    parser.add_argument('--subjects', type=int, nargs='+', default=[2, 5])
    parser.add_argument('--seed', type=int, default=13)
    parser.add_argument('--batch-size', type=int, default=64)
    args = parser.parse_args()
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    lock = (output/'.run.lock').open('a+b')
    if lock.tell() == 0:
        lock.write(b'0'); lock.flush()
    lock.seek(0)
    msvcrt.locking(lock.fileno(), msvcrt.LK_NBLCK, 1)
    try:
        signature = {'source_hashes': {name: hashlib.sha256((ROOT/name).read_bytes()).hexdigest()
            for name in SOURCES}, 'epochs': args.epochs, 'subjects': args.subjects,
            'seed': args.seed, 'batch_size': args.batch_size, 'arms': ARMS}
        if (output/'signature.json').exists():
            if json.loads((output/'signature.json').read_text()) != signature:
                raise RuntimeError('Source or schedule changed; use a new directory')
        write_json(output/'signature.json', signature)
        if not (output/'source').exists():
            for name in SOURCES:
                destination = output/'source'/name
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(ROOT/name, destination)
        if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
            raise RuntimeError('CUDA BF16 GPU is required')
        torch.set_num_threads(8)
        torch.set_num_interop_threads(2)
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        corpus = O2Corpus()
        rows = []
        jobs = [(subject, arm) for subject in args.subjects for arm in ARMS]
        for number, (subject, arm) in enumerate(jobs, 1):
            write_json(output/'status.json', {'state': 'running', 'updated': datetime.now().astimezone().isoformat(),
                'current_job': number, 'total_jobs': len(jobs), 'subject': subject, 'arm': arm[0]})
            fold = corpus.fold(subject, 'mag_dot')
            rows.append(run_one(fold, arm, subject, args.seed, args.epochs, args.batch_size, output))
            write_json(output/'summary.json', {'source_only': True,
                'rows': [{key: row[key] for key in ('arm', 'subject', 'seed', 'validation_subject',
                    'best_epoch', 'best_source_ideal_f1', 'source_self_f1_at_best', 'best_probe',
                    'last_probe', 'seconds')} for row in rows]})
            del fold
            torch.cuda.empty_cache()
        write_json(output/'completed.json', {'completed_jobs': len(jobs), 'expected_jobs': len(jobs),
            'source_only': True, 'finished': datetime.now().astimezone().isoformat()})
        write_json(output/'status.json', {'state': 'complete', 'completed_jobs': len(jobs),
            'total_jobs': len(jobs), 'updated': datetime.now().astimezone().isoformat()})
        print(json.dumps({'completed_jobs': len(jobs), 'output': str(output)}), flush=True)
    finally:
        lock.close()


if __name__ == '__main__':
    main()
