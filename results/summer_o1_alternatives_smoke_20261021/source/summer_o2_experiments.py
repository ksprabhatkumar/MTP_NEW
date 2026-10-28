"""Objective 2 feature experiments on the verified summer GIN-attention model.

This module is isolated from the active O1 source fingerprint. It is not run on
the GPU until O1 has completed and the O2 preflight/smoke have passed.
"""

from dataclasses import asdict, dataclass
import json
from pathlib import Path
import random
import time

import numpy as np
import torch
from torch.nn import functional as F

from models.summer_dynamic import SummerDynamicWHAR
from o2_features import WIDTHS, random_rotations, rotate_raw
from o2_experiments import batches, evaluate
from summer_o1_experiments import optimizer_for, routing_audit, scheduler_for
from workstation_experiments import Condition, metric_values, profile, save_checkpoint, seed_all, write_json


@dataclass(frozen=True)
class Variant:
    kind: str
    graph: str

    @property
    def name(self):
        return f'{self.kind}_{self.graph}'


VARIANTS = [Variant(kind, graph) for kind in ('magnitude', 'mag_dot')
            for graph in ('physical', 'residual', 'none')]
VARIANTS.append(Variant('raw_aug', 'none'))
BRIDGE = Variant('raw', 'physical')


def model_for(variant):
    return SummerDynamicWHAR(in_features=WIDTHS[variant.kind],
                             graph_variant=variant.graph,
                             use_reconstruction=False).cuda()


def train_epoch(model, optimizer, fold, variant, batch_size, limit=None):
    model.train()
    total, count = 0.0, 0
    started = time.perf_counter()
    mean, std = fold['mean'].cuda(), fold['std'].cuda()
    for batch in batches(fold['data']['train'], batch_size, shuffle=True, limit=limit):
        x, y, availability = batch['x'], batch['y'], batch['a']
        if variant.kind == 'raw_aug':
            rotated = rotate_raw(batch['raw'], random_rotations(
                len(x), x.shape[2], x.device))
            x = ((rotated - mean) / std) * availability[:, None, :, None]
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast('cuda', dtype=torch.bfloat16):
            logits, _ = model(x, availability)
            loss = F.cross_entropy(logits, y)
        if not torch.isfinite(loss):
            raise FloatingPointError('Nonfinite summer O2 loss')
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
        optimizer.step()
        total += float(loss) * len(y)
        count += len(y)
    torch.cuda.synchronize()
    return {'loss': total / count, 'windows': count,
            'seconds': time.perf_counter() - started}


def run_job(fold, variant, subject, seed, epochs, batch_size, output,
            limit=None, validation_only=False, model_factory=None,
            routing_fn=None):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    result_file = output / 'result.json'
    if result_file.exists():
        return json.loads(result_file.read_text())
    seed_all(seed)
    model = model_for(variant) if model_factory is None else model_factory(variant)
    optimizer = optimizer_for(model)
    scheduler = scheduler_for(optimizer, epochs) if limit is None else None
    config = {'variant': asdict(variant), 'subject': subject, 'seed': seed,
              'epochs': epochs, 'batch_size': batch_size, 'smoke_limit': limit,
              'validation_only': validation_only, 'hidden_dim': 16,
              'dropout': .5, 'optimizer': 'Adam', 'lr': 1e-4,
              'weight_decay': 5e-4,
              'scheduler': 'OneCycleLR once/epoch' if scheduler else 'constant for smoke',
              'selection': 'distinct source subject ideal 33-class macro-F1',
              'loss': 'cross_entropy', 'backbone': 'verified summer GIN-attention',
              'feature_width': WIDTHS[variant.kind]}
    write_json(output / 'config.json', config)
    first, best, best_epoch, history = 1, -1.0, 0, []
    latest = output / 'latest.pt'
    if latest.exists():
        state = torch.load(latest, map_location='cpu', weights_only=False)
        if state['config'] != config:
            raise ValueError('Summer O2 checkpoint configuration changed')
        model.load_state_dict(state['model'])
        optimizer.load_state_dict(state['optimizer'])
        if scheduler:
            scheduler.load_state_dict(state['scheduler'])
        first, best = state['epoch'] + 1, state['best']
        best_epoch, history = state['best_epoch'], state['history']
        torch.set_rng_state(state['cpu_rng'])
        torch.cuda.set_rng_state_all(state['cuda_rng'])
        random.setstate(state['python_rng'])
        np.random.set_state(state['numpy_rng'])
    started = time.perf_counter()
    with (output / 'epochs.jsonl').open('a', encoding='utf-8', buffering=1) as log:
        for epoch in range(first, epochs + 1):
            train = train_epoch(model, optimizer, fold, variant, batch_size, limit)
            validation, _, _ = evaluate(model, fold, 'val_ideal', batch_size,
                                         limit=limit)
            row = {'epoch': epoch, 'train': train,
                   'validation_ideal_f1': validation['macro_f1'],
                   'lr': optimizer.param_groups[0]['lr']}
            history.append(row)
            log.write(json.dumps(row) + '\n')
            if validation['macro_f1'] > best:
                best, best_epoch = validation['macro_f1'], epoch
                save_checkpoint(output / 'best.pt',
                    {'model': model.state_dict(), 'config': config,
                     'epoch': epoch, 'validation_f1': best,
                     'mean': fold['mean'], 'std': fold['std']})
            if scheduler:
                scheduler.step()
            save_checkpoint(latest,
                {'model': model.state_dict(), 'optimizer': optimizer.state_dict(),
                 'scheduler': scheduler.state_dict() if scheduler else None,
                 'config': config, 'epoch': epoch, 'best': best,
                 'best_epoch': best_epoch, 'history': history,
                 'cpu_rng': torch.get_rng_state(),
                 'cuda_rng': torch.cuda.get_rng_state_all(),
                 'python_rng': random.getstate(), 'numpy_rng': np.random.get_state()})
            print(f'{variant.name} S{subject:02d} seed={seed} '
                  f'epoch={epoch}/{epochs} loss={train["loss"]:.4f} '
                  f'source_val_F1={validation["macro_f1"]:.4f}', flush=True)
    model.load_state_dict(torch.load(output / 'best.pt', map_location='cpu',
                                      weights_only=False)['model'])
    result = {'condition': variant.name, 'kind': variant.kind,
              'graph': variant.graph, 'subject': subject, 'seed': seed,
              'best_epoch': best_epoch, 'best_validation_f1': best,
              'epochs': epochs, 'history': history, 'config': config,
              'split': fold['summary'], 'metrics': {}}
    scenarios = ('val_ideal', 'val_self') if validation_only else (
        'ideal', 'self', 'induced')
    for scenario in scenarios:
        if scenario not in fold['data']:
            continue
        metrics, y, pred = evaluate(model, fold, scenario, batch_size, limit=limit)
        result['metrics'][scenario] = metrics
        np.savez_compressed(output / f'predictions_{scenario}.npz',
            labels=y, predictions=pred,
            corpus_indices=fold['indices'][scenario][:len(y)])
        if 'sources' in fold:
            sources = fold['sources'][scenario][:len(y)]
            result.setdefault('per_recording', {})[scenario] = {
                source: {
                    'fixed33': metric_values(y[sources == source],
                                              pred[sources == source]),
                    'supported': metric_values(y[sources == source],
                                                pred[sources == source],
                                                np.unique(y[sources == source]).tolist())}
                for source in np.unique(sources)}
    if not validation_only:
        rotated_scores = []
        for repeat in range(3):
            metrics, y, pred = evaluate(model, fold, 'ideal', batch_size,
                rotated=True, rotation_seed=2026 + subject * 101 + repeat,
                limit=limit)
            rotated_scores.append(metrics)
            np.savez_compressed(output / f'predictions_rotated_{repeat}.npz',
                labels=y, predictions=pred,
                corpus_indices=fold['indices']['ideal'][:len(y)])
        result['rotated_ideal'] = rotated_scores
        result['routing'] = (routing_audit(model, fold, variant) if routing_fn is None
                             else routing_fn(model, fold, variant))
        ideal = fold['data']['ideal']
        result['efficiency'] = profile(model,
            (ideal['x'], ideal['y'], ideal['a']),
            Condition(variant.name, variant.graph))
    result['seconds_this_invocation'] = time.perf_counter() - started
    write_json(result_file, result)
    del model, optimizer
    torch.cuda.empty_cache()
    return result


def summarize(output):
    rows = [json.loads(path.read_text()) for path in
            (Path(output) / 'jobs').glob('*/*/*/result.json')]
    summary = {}
    for variant in VARIANTS:
        selected = [r for r in rows if r['condition'] == variant.name]
        entry = {'completed_jobs': len(selected), 'expected_jobs': 9}
        for scenario in ('ideal', 'self', 'induced'):
            subject_means = [np.mean([r['metrics'][scenario]['macro_f1']
                for r in selected if r['subject'] == subject])
                for subject in (2, 5, 15)
                if any(r['subject'] == subject for r in selected)]
            if subject_means:
                entry[scenario] = {'subject_mean': float(np.mean(subject_means)),
                    'subject_std': float(np.std(subject_means, ddof=1))
                    if len(subject_means) > 1 else None}
        summary[variant.name] = entry
    write_json(Path(output) / 'summary.json', summary)
