"""Matched rotation-feature experiments with validation-only model selection."""
from dataclasses import dataclass, asdict
import csv
import json
from pathlib import Path
import random
import time

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import f1_score, confusion_matrix

from models.mamba_whar import GAE_Mamba_WHAR
from o2_features import WIDTHS, features, random_rotations, rotate_raw
from workstation_experiments import (metric_values, profile, routing_artifacts,
    save_checkpoint, seed_all, write_json, Condition)


@dataclass(frozen=True)
class Variant:
    kind: str
    graph: str

    @property
    def name(self):
        return f'{self.kind}_{self.graph}'


VARIANTS = [Variant(kind, graph) for kind in ('raw', 'raw_aug', 'magnitude', 'mag_dot')
            for graph in ('none', 'residual')]


def model_for(variant):
    return GAE_Mamba_WHAR(in_features=WIDTHS[variant.kind],
                          graph_variant=variant.graph).cuda()


def batches(part, batch_size, shuffle=False, limit=None):
    count = len(part['y'])
    indices = torch.randperm(count, device='cuda') if shuffle else torch.arange(count, device='cuda')
    for number, start in enumerate(range(0, count, batch_size)):
        if limit is not None and number >= limit:
            break
        take = indices[start:start + batch_size]
        yield {key: tensor[take] for key, tensor in part.items()}


def train_epoch(model, optimizer, fold, variant, batch_size, limit=None):
    model.train()
    total, correct, count = 0., 0, 0
    started = time.perf_counter()
    mean, std = fold['mean'].cuda(), fold['std'].cuda()
    for batch in batches(fold['data']['train'], batch_size, shuffle=True, limit=limit):
        x = batch['x']
        if variant.kind == 'raw_aug':
            raw = rotate_raw(batch['raw'], random_rotations(len(x), x.shape[2], x.device))
            x = ((raw - mean) / std) * batch['a'][:, None, :, None]
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast('cuda', dtype=torch.bfloat16):
            logits, _ = model(x, batch['a'])
            loss = F.cross_entropy(logits, batch['y'])
        if not torch.isfinite(loss):
            raise FloatingPointError('Nonfinite O2 training loss')
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
        optimizer.step()
        total += float(loss) * len(x)
        correct += int((logits.argmax(-1) == batch['y']).sum())
        count += len(x)
    torch.cuda.synchronize()
    return {'loss': total / count, 'accuracy': correct / count,
            'seconds': time.perf_counter() - started, 'windows': count}


@torch.inference_mode()
def evaluate(model, fold, part_name, batch_size, rotated=False, rotation_seed=0, limit=None):
    model.eval()
    part = fold['data'][part_name]
    mean, std = fold['mean'].cuda(), fold['std'].cuda()
    generator = torch.Generator(device='cuda').manual_seed(rotation_seed) if rotated else None
    y_list, prediction_list, total_loss, count = [], [], 0., 0
    for batch in batches(part, batch_size, limit=limit):
        x = batch['x']
        if rotated:
            raw = rotate_raw(batch['raw'], random_rotations(len(x), x.shape[2], x.device, generator))
            x = ((features(raw, fold['summary']['feature_mode']) - mean) / std) * batch['a'][:, None, :, None]
        with torch.autocast('cuda', dtype=torch.bfloat16):
            logits, _ = model(x, batch['a'])
            loss = F.cross_entropy(logits, batch['y'], reduction='sum')
        if not torch.isfinite(logits).all():
            raise FloatingPointError('Nonfinite O2 logits')
        total_loss += float(loss)
        count += len(x)
        y_list.append(batch['y'].cpu().numpy())
        prediction_list.append(logits.argmax(-1).cpu().numpy())
    y = np.concatenate(y_list)
    prediction = np.concatenate(prediction_list)
    metrics = metric_values(y, prediction)
    metrics['loss'] = total_loss / count
    present = sorted(set(y.tolist()))
    metrics['supported_class_macro_f1'] = float(f1_score(
        y, prediction, labels=present, average='macro', zero_division=0))
    return metrics, y, prediction


def validation_selection(ideal, self_result, common):
    # Restrict true-label rows to shared classes; predictions outside the shared
    # set remain errors. This rule is frozen before O2 test evaluation.
    vals = []
    for _, y, prediction in (ideal, self_result):
        keep = np.isin(y, common)
        vals.append(float(f1_score(y[keep], prediction[keep], labels=common,
                                    average='macro', zero_division=0)))
    ideal_f1, self_f1 = vals
    score = .5 * (ideal_f1 + self_f1) - 2 * max(0, .85 - ideal_f1)
    return {'score': score, 'ideal_common_f1': ideal_f1,
            'self_common_f1': self_f1, 'ideal_floor': .85}


def optimizer_for(model):
    return torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=5e-4, fused=True)


def run_job(fold, variant, subject, seed, epochs, batch_size, output, limit=None):
    output.mkdir(parents=True, exist_ok=True)
    completed = output / 'result.json'
    if completed.exists():
        return json.loads(completed.read_text())
    seed_all(seed)
    model = model_for(variant)
    optimizer = optimizer_for(model)
    config = {'variant': asdict(variant), 'subject': subject, 'seed': seed,
        'epochs': epochs, 'batch_size': batch_size, 'smoke_batch_limit': limit,
        'lr': 1e-4, 'weight_decay': 5e-4,
        'checkpoint_rule': 'mean ideal/self common-class macro-F1 - 2*max(0,0.85-ideal)',
        'training_scenarios': ['ideal'], 'precision': 'bf16'}
    write_json(output / 'config.json', config)
    history, best_score, best_epoch, first_epoch = [], float('-inf'), 0, 1
    latest = output / 'latest.pt'
    if latest.exists():
        state = torch.load(latest, map_location='cpu', weights_only=False)
        if state['config'] != config:
            raise ValueError('Checkpoint configuration mismatch')
        model.load_state_dict(state['model'])
        optimizer.load_state_dict(state['optimizer'])
        history, best_score, best_epoch = state['history'], state['best_score'], state['best_epoch']
        first_epoch = state['epoch'] + 1
        torch.set_rng_state(state['cpu_rng'])
        torch.cuda.set_rng_state_all(state['cuda_rng'])
        random.setstate(state['python_rng'])
        np.random.set_state(state['numpy_rng'])
        del state
    started = time.perf_counter()
    common = fold['summary']['common_validation_classes']
    with (output / 'epochs.jsonl').open('a', encoding='utf-8', buffering=1) as log:
        for epoch in range(first_epoch, epochs + 1):
            train = train_epoch(model, optimizer, fold, variant, batch_size, limit)
            val_ideal = evaluate(model, fold, 'val_ideal', batch_size, limit=limit)
            val_self = evaluate(model, fold, 'val_self', batch_size, limit=limit)
            selection = validation_selection(val_ideal, val_self, common)
            entry = {'epoch': epoch, 'train': train, 'selection': selection,
                     'elapsed_seconds': time.perf_counter() - started}
            history.append(entry)
            log.write(json.dumps(entry) + '\n')
            if selection['score'] > best_score:
                best_score, best_epoch = selection['score'], epoch
                save_checkpoint(output / 'best.pt', {'model': model.state_dict(),
                    'epoch': epoch, 'score': best_score, 'config': config,
                    'mean': fold['mean'], 'std': fold['std']})
            save_checkpoint(latest, {'model': model.state_dict(),
                'optimizer': optimizer.state_dict(), 'epoch': epoch,
                'best_score': best_score, 'best_epoch': best_epoch,
                'history': history, 'config': config, 'cpu_rng': torch.get_rng_state(),
                'cuda_rng': torch.cuda.get_rng_state_all(),
                'python_rng': random.getstate(), 'numpy_rng': np.random.get_state()})
            print(f'{variant.name} S{subject:02d} seed={seed} epoch={epoch}/{epochs} '
                  f'loss={train["loss"]:.4f} val_ideal={selection["ideal_common_f1"]:.4f} '
                  f'val_self={selection["self_common_f1"]:.4f} select={best_score:.4f}', flush=True)
    model.load_state_dict(torch.load(output / 'best.pt', map_location='cpu', weights_only=False)['model'])
    result = {'condition': variant.name, 'kind': variant.kind, 'graph': variant.graph,
              'subject': subject, 'seed': seed, 'best_epoch': best_epoch,
              'best_validation_score': best_score, 'epochs': epochs,
              'split': fold['summary'], 'metrics': {}, 'history': history}
    for scenario in ('ideal', 'self', 'induced'):
        if scenario not in fold['data']:
            continue
        metrics, y, prediction = evaluate(model, fold, scenario, batch_size, limit=limit)
        result['metrics'][scenario] = metrics
        np.savez_compressed(output / f'predictions_{scenario}.npz',
            labels=y, predictions=prediction,
            corpus_indices=fold['indices'][scenario][:len(y)])
        np.savetxt(output / f'confusion_{scenario}.csv',
                   confusion_matrix(y, prediction, labels=range(33)), fmt='%d', delimiter=',')
    # Fixed synthetic rotations, never used to select the checkpoint.
    rotations = []
    for repeat in range(3):
        metrics, y, prediction = evaluate(model, fold, 'ideal', batch_size,
            rotated=True, rotation_seed=2026 + subject * 101 + repeat, limit=limit)
        rotations.append(metrics)
        np.savez_compressed(output / f'predictions_rotated_{repeat}.npz', labels=y,
            predictions=prediction, corpus_indices=fold['indices']['ideal'][:len(y)])
    result['rotated_ideal'] = rotations
    result['efficiency'] = profile(model,
        (fold['data']['ideal']['x'], fold['data']['ideal']['y'], fold['data']['ideal']['a']),
        Condition(variant.name, variant.graph))
    if variant.graph == 'residual':
        routing_artifacts(model, {'data': {key: (part['x'], part['y'], part['a'])
            for key, part in fold['data'].items() if key in ('ideal','self','induced')}},
            Condition(variant.name, variant.graph), output)
    result['job_seconds_this_invocation'] = time.perf_counter() - started
    write_json(completed, result)
    del model, optimizer
    torch.cuda.empty_cache()
    return result


def summarize(output):
    results = [json.loads(path.read_text()) for path in (output / 'jobs').glob('*/*/*/result.json')]
    rows = []
    for result in results:
        row = {key: result[key] for key in ('condition','kind','graph','subject','seed','best_epoch')}
        for scenario in ('ideal','self','induced'):
            metrics = result['metrics'].get(scenario, {})
            row[f'{scenario}_macro_f1_33'] = metrics.get('macro_f1')
            row[f'{scenario}_supported_f1'] = metrics.get('supported_class_macro_f1')
        row['rotated_ideal_macro_f1_33'] = np.mean(
            [item['macro_f1'] for item in result['rotated_ideal']])
        row['batch_one_median_ms'] = result['efficiency']['batch_one_median_ms']
        rows.append(row)
    if rows:
        with (output / 'fold_metrics.csv').open('w', newline='', encoding='utf-8') as file:
            writer = csv.DictWriter(file, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    summary = {}
    for variant in VARIANTS:
        selected = [r for r in rows if r['condition'] == variant.name and r['subject'] in (2,5,15)]
        report = {'completed_primary_jobs': len(selected), 'expected_primary_jobs': 9}
        for metric in ('ideal_macro_f1_33','self_macro_f1_33',
                       'induced_macro_f1_33','rotated_ideal_macro_f1_33'):
            means = [np.mean([r[metric] for r in selected if r['subject'] == subject and r[metric] is not None])
                     for subject in (2,5,15) if any(r['subject']==subject and r[metric] is not None for r in selected)]
            if means:
                report[metric] = {'subject_mean': float(np.mean(means)),
                    'subject_std': float(np.std(means, ddof=1)) if len(means)>1 else None}
        summary[variant.name] = report
    write_json(output / 'summary.json', summary)
