"""GPU-resident, leakage-controlled O1 experiments and durable result artifacts."""
from contextlib import nullcontext
import csv
from dataclasses import dataclass, asdict
import json
import math
from pathlib import Path
import random
import time

import numpy as np
import torch
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_recall_fscore_support

from models.mamba_whar import GAE_Mamba_WHAR
from prepare_workstation_data import CACHE


@dataclass(frozen=True)
class Condition:
    name: str
    graph: str
    dropout: float = 0.0
    gae: bool = False
    sender_mask: bool = True


CONDITIONS = [Condition('none', 'none'), Condition('physical', 'physical'),
              Condition('global', 'global'), Condition('dynamic', 'dynamic'),
              Condition('residual', 'residual'),
              Condition('physical_dropout', 'physical', .1),
              Condition('residual_dropout', 'residual', .1),
              Condition('residual_gae', 'residual', .1, True),
              Condition('residual_unmasked', 'residual', .1, False, False),
              Condition('dynamicwhar', 'dynamicwhar')]


def write_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False), encoding='utf-8')
    temporary.replace(path)


def save_checkpoint(path, value):
    temporary = path.with_suffix('.tmp')
    torch.save(value, temporary)
    temporary.replace(path)


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def amp_context():
    return torch.autocast('cuda', dtype=torch.bfloat16)


def make_model(condition):
    if condition.graph == 'dynamicwhar':
        from baselines.dynamicwhar_adapter import Objective1DynamicWHAR
        return Objective1DynamicWHAR().cuda()
    return GAE_Mamba_WHAR(graph_variant=condition.graph, use_reconstruction=condition.gae).cuda()


class Corpus:
    def __init__(self):
        self.audit = json.loads((CACHE / 'audit.json').read_text())
        arrays = {k: [] for k in ('x', 'y', 'availability', 'starts')}
        self.subjects, self.scenarios, self.sources = [], [], []
        for row in self.audit['recordings']:
            with np.load(CACHE / (Path(row['source']).stem + '.npz')) as data:
                if len(data['y']) != row['windows'] or data['x'].shape[1:] != (50, 9, 6):
                    raise ValueError(f"Cache/manifest mismatch: {row['source']}")
                for key in arrays:
                    arrays[key].append(data[key])
                count = len(data['y'])
            self.subjects.extend([row['subject']] * count)
            self.scenarios.extend([row['scenario']] * count)
            self.sources.extend([row['source']] * count)
        for key, chunks in arrays.items():
            setattr(self, key, torch.from_numpy(np.concatenate(chunks)))
        self.subjects = np.asarray(self.subjects)
        self.scenarios = np.asarray(self.scenarios)
        self.sources = np.asarray(self.sources)
        if not torch.isfinite(self.x).all():
            raise ValueError('Nonfinite input corpus')

    def fold(self, subject):
        validation = subject % 17 + 1
        ideal = self.scenarios == 'ideal'
        masks = {'train': ideal & ~np.isin(self.subjects, [subject, validation]),
                 'validation': ideal & (self.subjects == validation),
                 'ideal': ideal & (self.subjects == subject),
                 'induced': (self.scenarios == 'induced') & (self.subjects == subject),
                 'self': (self.scenarios == 'self') & (self.subjects == subject)}
        indices = {key: np.flatnonzero(mask) for key, mask in masks.items()}
        for key in ('train', 'validation', 'ideal'):
            if not len(indices[key]):
                raise ValueError(f'Empty {key} partition for subject {subject}')
        train = self.x[indices['train']]
        mean = train.mean(dim=(0, 1), keepdim=True)
        std = train.std(dim=(0, 1), keepdim=True).clamp_min(1e-6)
        del train
        data = {}
        for key, ix in indices.items():
            if not len(ix):
                continue
            a = self.availability[ix]
            x = ((self.x[ix] - mean) / std) * a[:, None, :, None]
            data[key] = (x.cuda(), self.y[ix].cuda(), a.cuda())
        summary = {'test_subject': subject, 'validation_subject': validation,
                   'training_subjects': sorted(set(self.subjects[indices['train']].tolist())),
                   'counts': {key: len(ix) for key, ix in indices.items()},
                   'recordings': {key: sorted(set(self.sources[ix])) for key, ix in indices.items()},
                   'class_counts': {key: np.bincount(self.y[ix].numpy(), minlength=33).tolist()
                                    for key, ix in indices.items()},
                   'normalization_axes': 'windows,time; separate mean/std per sensor/channel'}
        return {'data': data, 'indices': indices, 'mean': mean, 'std': std, 'summary': summary}


def batches(tensors, batch_size, shuffle=False, limit=None):
    count = len(tensors[1])
    order = torch.randperm(count, device='cuda') if shuffle else torch.arange(count, device='cuda')
    for batch_id, start in enumerate(range(0, count, batch_size)):
        if limit is not None and batch_id >= limit:
            break
        index = order[start:start + batch_size]
        yield tuple(t[index] for t in tensors)


def drop_sensor(x, availability, probability):
    mask = availability.clone()
    if probability:
        # One random available sensor removed in the selected windows; no CPU loops.
        scores = torch.rand(mask.shape, device=x.device).masked_fill(~mask, -1)
        sensor = scores.argmax(dim=1)
        rows = torch.arange(len(x), device=x.device)
        selected = torch.rand(len(x), device=x.device) < probability
        mask[rows[selected], sensor[selected]] = False
    return x * mask[:, None, :, None], mask


def metric_values(labels, predictions, classes=None):
    classes = list(range(33)) if classes is None else list(classes)
    if not len(labels):
        return None
    precision, recall, f1, support = precision_recall_fscore_support(
        labels, predictions, labels=classes, zero_division=0)
    return {'windows': len(labels), 'accuracy': float(accuracy_score(labels, predictions)),
            'macro_f1': float(f1.mean()),
            'weighted_f1': float(f1_score(labels, predictions, average='weighted', zero_division=0)),
            'macro_classes': classes,
            'per_class': [{'label': int(c), 'precision': float(p), 'recall': float(r),
                           'f1': float(f), 'support': int(n)}
                          for c, p, r, f, n in zip(classes, precision, recall, f1, support)]}


@torch.inference_mode()
def evaluate(model, tensors, condition, batch_size, limit=None, missing_sensor=None):
    model.eval()
    labels, predictions = [], []
    total_loss, count = 0., 0
    for x, y, a in batches(tensors, batch_size, limit=limit):
        if missing_sensor is not None:
            a = a.clone()
            a[:, missing_sensor] = False
            x = x * a[:, None, :, None]
        with amp_context():
            logits, _ = model(x, a if condition.sender_mask else None)
            loss = torch.nn.functional.cross_entropy(logits, y, reduction='sum')
        if not torch.isfinite(logits).all():
            raise FloatingPointError('Nonfinite evaluation logits')
        total_loss += loss.item()
        count += len(y)
        labels.append(y.cpu().numpy())
        predictions.append(logits.argmax(1).cpu().numpy())
    labels, predictions = np.concatenate(labels), np.concatenate(predictions)
    metrics = metric_values(labels, predictions)
    metrics['loss'] = total_loss / count
    return metrics, labels, predictions


def train_epoch(model, optimizer, tensors, condition, batch_size, limit=None):
    model.train()
    total_loss, correct, count = 0., 0, 0
    started = time.perf_counter()
    for clean, y, a in batches(tensors, batch_size, shuffle=True, limit=limit):
        x, masked_a = drop_sensor(clean, a, condition.dropout)
        optimizer.zero_grad(set_to_none=True)
        with amp_context():
            logits, reconstruction = model(x, masked_a if condition.sender_mask else None)
            loss = torch.nn.functional.cross_entropy(logits, y)
            if condition.gae:
                hidden = (a & ~masked_a)[:, None, :, None].expand_as(clean)
                # Avoid NaN on batches with no intentionally hidden targets.
                error = (reconstruction.float() - clean).square() * hidden
                loss = loss + .01 * error.sum() / hidden.sum().clamp_min(1)
        if not torch.isfinite(loss):
            raise FloatingPointError('Nonfinite training loss')
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
        optimizer.step()
        total_loss += loss.item() * len(y)
        correct += (logits.argmax(1) == y).sum().item()
        count += len(y)
    torch.cuda.synchronize()
    return {'loss': total_loss / count, 'accuracy': correct / count,
            'seconds': time.perf_counter() - started, 'windows': count}


def optimizer_for(model):
    return torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=5e-4, fused=True)


@torch.inference_mode()
def profile(model, tensors, condition):
    x, _, a = [t[:1] for t in tensors]
    model.eval()
    timings = []
    with amp_context():
        for _ in range(8):
            model(x, a if condition.sender_mask else None)
        torch.cuda.synchronize()
        baseline = torch.cuda.memory_allocated()
        torch.cuda.reset_peak_memory_stats()
        for _ in range(40):
            started = time.perf_counter()
            model(x, a if condition.sender_mask else None)
            torch.cuda.synchronize()
            timings.append((time.perf_counter() - started) * 1000)
    return {'parameters': sum(p.numel() for p in model.parameters()),
            'batch_one_median_ms': float(np.median(timings)),
            'batch_one_p95_ms': float(np.percentile(timings, 95)),
            'batch_one_incremental_peak_bytes': torch.cuda.max_memory_allocated() - baseline,
            'gpu': torch.cuda.get_device_name(), 'precision': 'bfloat16 autocast',
            'includes': 'encoder, graph construction, temporal block and classifier',
            'excludes': 'disk IO and offline normalization'}


def heatmap_svg(matrix, path, title):
    cells = []
    for i, row in enumerate(matrix):
        for j, value in enumerate(row):
            v = float(value)
            cells.append(f'<rect x="{60+j*48}" y="{55+i*48}" width="48" height="48" '
                         f'fill="rgb({int(255*(1-v))},{int(255*(1-v))},255)" stroke="white"/>')
            cells.append(f'<text x="{84+j*48}" y="{83+i*48}" text-anchor="middle" font-size="11">{v:.2f}</text>')
    for i in range(9):
        cells.append(f'<text x="{84+i*48}" y="45" text-anchor="middle">{i}</text>')
        cells.append(f'<text x="40" y="{83+i*48}">{i}</text>')
    path.write_text('<svg xmlns="http://www.w3.org/2000/svg" width="550" height="530">'
                    '<rect width="100%" height="100%" fill="white"/>'
                    f'<text x="20" y="22">{title}</text>' + ''.join(cells) +
                    '<text x="60" y="515">Rows: receivers; columns: senders (node index)</text></svg>')


@torch.inference_mode()
def routing_artifacts(model, fold, condition, output):
    if condition.graph == 'dynamicwhar':
        return
    routing = {}
    for scenario in ('ideal', 'induced', 'self'):
        if scenario not in fold['data']:
            continue
        x, _, a = (t[:128] for t in fold['data'][scenario])
        model.eval()
        with amp_context():
            _, _, diagnostics = model(x, a if condition.sender_mask else None, return_diagnostics=True)
        avg = diagnostics['adjacency'].float().mean((0, 1)).cpu().numpy()
        routing[scenario] = {'mean_adjacency': avg.tolist(),
                             'branch_weights': diagnostics['branch_weights'].float().cpu().tolist(),
                             'sample_windows': len(x), 'sampling': 'first 128 scenario windows; descriptive only'}
        heatmap_svg(avg, output / f'routing_{scenario}.svg', f'{condition.name}: {scenario}')
    write_json(output / 'routing.json', routing)


def train_job(fold, condition, subject, seed, epochs, batch_size, output, limit=None):
    output.mkdir(parents=True, exist_ok=True)
    result_path = output / 'result.json'
    if result_path.exists():
        return json.loads(result_path.read_text())
    seed_all(seed)
    model = make_model(condition)
    optimizer = optimizer_for(model)
    config = {'condition': asdict(condition), 'subject': subject, 'seed': seed,
              'epochs': epochs, 'batch_size': batch_size, 'smoke_batch_limit': limit,
              'learning_rate': 1e-4, 'weight_decay': 5e-4, 'reconstruction_weight': .01,
              'precision': 'bf16', 'backbone': 'inherited gated causal convolution, not selective-scan Mamba'}
    write_json(output / 'config.json', config)
    history, best_f1, best_epoch, first_epoch = [], -1., 0, 1
    latest = output / 'latest.pt'
    if latest.exists():
        checkpoint = torch.load(latest, map_location='cpu', weights_only=False)
        if checkpoint['config'] != config:
            raise ValueError('Checkpoint configuration mismatch')
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        history, best_f1, best_epoch = checkpoint['history'], checkpoint['best_f1'], checkpoint['best_epoch']
        first_epoch = checkpoint['epoch'] + 1
        torch.set_rng_state(checkpoint['cpu_rng'])
        torch.cuda.set_rng_state_all(checkpoint['cuda_rng'])
        random.setstate(checkpoint['python_rng'])
        np.random.set_state(checkpoint['numpy_rng'])
        del checkpoint
    started = time.perf_counter()
    torch.cuda.reset_peak_memory_stats()
    with (output / 'epochs.jsonl').open('a', encoding='utf-8', buffering=1) as log:
        for epoch in range(first_epoch, epochs + 1):
            train = train_epoch(model, optimizer, fold['data']['train'], condition, batch_size, limit)
            val, _, _ = evaluate(model, fold['data']['validation'], condition, batch_size, limit)
            row = {'epoch': epoch, 'train': train, 'validation_macro_f1': val['macro_f1'],
                   'validation_loss': val['loss'], 'elapsed_seconds': time.perf_counter() - started}
            history.append(row)
            log.write(json.dumps(row) + '\n')
            if val['macro_f1'] > best_f1:
                best_f1, best_epoch = val['macro_f1'], epoch
                save_checkpoint(output / 'best.pt', {'model': model.state_dict(), 'epoch': epoch,
                    'validation_macro_f1': best_f1, 'config': config,
                    'mean': fold['mean'], 'std': fold['std']})
            # Persist every epoch; no loss of completed epoch work on restart.
            save_checkpoint(latest, {'model': model.state_dict(), 'optimizer': optimizer.state_dict(),
                'epoch': epoch, 'best_f1': best_f1, 'best_epoch': best_epoch, 'history': history,
                'config': config, 'cpu_rng': torch.get_rng_state(),
                'cuda_rng': torch.cuda.get_rng_state_all(), 'python_rng': random.getstate(),
                'numpy_rng': np.random.get_state()})
            print(f'{condition.name} S{subject:02d} seed={seed} epoch={epoch}/{epochs} '
                  f'loss={train["loss"]:.4f} val_F1={val["macro_f1"]:.4f} '
                  f'train={train["seconds"]:.2f}s', flush=True)
    train_peak = torch.cuda.max_memory_allocated()
    model.load_state_dict(torch.load(output / 'best.pt', map_location='cpu', weights_only=False)['model'])
    result = {'condition': condition.name, 'subject': subject, 'seed': seed,
              'epochs': epochs, 'best_epoch': best_epoch, 'validation_macro_f1': best_f1,
              'split': fold['summary'], 'metrics': {}, 'history': history,
              'training_peak_allocated_bytes_including_data': train_peak}
    predictions = {}
    for scenario in ('ideal', 'induced', 'self'):
        if scenario not in fold['data']:
            continue
        metrics, y, pred = evaluate(model, fold['data'][scenario], condition, batch_size, limit)
        result['metrics'][scenario] = metrics
        predictions[scenario] = (y, pred)
        np.savez_compressed(output / f'predictions_{scenario}.npz', labels=y, predictions=pred,
                            corpus_indices=fold['indices'][scenario][:len(y)])
        np.savetxt(output / f'confusion_{scenario}.csv', confusion_matrix(y, pred, labels=range(33)),
                   fmt='%d', delimiter=',')
    if 'induced' in predictions:
        iy, ip = predictions['ideal']
        dy, dp = predictions['induced']
        common = sorted(set(iy.tolist()) & set(dy.tolist()))
        imask, dmask = np.isin(iy, common), np.isin(dy, common)
        ideal_matched = metric_values(iy[imask], ip[imask], common)
        induced_matched = metric_values(dy[dmask], dp[dmask], common)
        result['matched_placement'] = {'classes': common, 'ideal': ideal_matched, 'induced': induced_matched,
            'degradation_macro_f1': ideal_matched['macro_f1'] - induced_matched['macro_f1'],
            'pairing': 'same subject and supported classes; recordings not time aligned'}
    result['sensor_dropout'] = {}
    for sensor in range(9):
        metrics, _, _ = evaluate(model, fold['data']['ideal'], condition, batch_size, limit, sensor)
        result['sensor_dropout'][str(sensor)] = {'macro_f1': metrics['macro_f1'], 'accuracy': metrics['accuracy']}
    result['efficiency'] = profile(model, fold['data']['ideal'], condition)
    routing_artifacts(model, fold, condition, output)
    result['job_seconds_this_invocation'] = time.perf_counter() - started
    write_json(result_path, result)
    del optimizer, model
    torch.cuda.empty_cache()
    return result


def summarize(output):
    results = [json.loads(p.read_text()) for p in (output / 'jobs').glob('*/*/*/result.json')]
    rows = []
    for r in results:
        row = {k: r[k] for k in ('condition', 'subject', 'seed', 'epochs', 'best_epoch')}
        row.update({f'{scenario}_{metric}': r['metrics'].get(scenario, {}).get(metric)
                    for scenario in ('ideal', 'induced', 'self') for metric in ('macro_f1', 'accuracy', 'weighted_f1')})
        row['matched_degradation'] = r.get('matched_placement', {}).get('degradation_macro_f1')
        rows.append(row)
    if rows:
        with (output / 'fold_metrics.csv').open('w', newline='') as file:
            writer = csv.DictWriter(file, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    summary = {}
    for c in CONDITIONS:
        selected = [r for r in rows if r['condition'] == c.name and r['subject'] in (2, 5, 15)]
        entry = {'completed_primary_jobs': len(selected), 'expected_primary_jobs': 9}
        for metric in ('ideal_macro_f1', 'induced_macro_f1', 'matched_degradation'):
            means = [np.mean([r[metric] for r in selected if r['subject'] == s and r[metric] is not None])
                     for s in (2, 5, 15) if any(r['subject'] == s and r[metric] is not None for r in selected)]
            if means:
                entry[metric] = {'subject_mean': float(np.mean(means)),
                                 'subject_std': float(np.std(means, ddof=1)) if len(means) > 1 else None,
                                 'worst_subject': float(min(means)) if metric != 'matched_degradation' else float(max(means))}
        summary[c.name] = entry
    write_json(output / 'summary.json', summary)
    return summary
