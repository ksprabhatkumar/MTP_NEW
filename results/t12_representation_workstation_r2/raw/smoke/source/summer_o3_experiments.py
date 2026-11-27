"""Sequential subject-client training on the summer graph architecture.

All model arms and method budgets are locked by the calling O3 session. This
module does not launch training on import.
"""

from dataclasses import asdict, dataclass
import json
from pathlib import Path
import random
import time

import numpy as np
import torch

from o2_features import features
from o2_experiments import evaluate
from summer_o1_experiments import optimizer_for, scheduler_for
from summer_o2_experiments import Variant, train_epoch
from summer_o3_federation import (FederatedSummerModel, apply_local_state,
    local_normalization_keys, round_communication_bytes,
    weighted_state_average)
from workstation_experiments import metric_values, save_checkpoint, seed_all, write_json


@dataclass(frozen=True)
class O3Job:
    kind: str
    graph: str
    method: str
    subject: int
    seed: int

    @property
    def arm(self):
        return f'{self.kind}_{self.graph}'


METHODS = ('centralized', 'local_only', 'fedavg', 'fedbn')


def clone_state(state):
    return {key: value.detach().clone() for key, value in state.items()}


def rng_state():
    return {'cpu': torch.get_rng_state(), 'cuda': torch.cuda.get_rng_state_all(),
            'python': random.getstate(), 'numpy': np.random.get_state()}


def restore_rng(state):
    torch.set_rng_state(state['cpu'])
    torch.cuda.set_rng_state_all(state['cuda'])
    random.setstate(state['python'])
    np.random.set_state(state['numpy'])


def split_clients(corpus, fold):
    indices = fold['indices']['train']
    subjects = corpus.subjects[indices]
    clients = {}
    for subject in fold['summary']['train_subjects']:
        positions = np.flatnonzero(subjects == subject)
        if not len(positions):
            raise ValueError(f'Empty O3 client {subject}')
        take = torch.as_tensor(positions, device='cuda')
        clients[subject] = {key: value[take]
                            for key, value in fold['data']['train'].items()}
    if sum(len(part['y']) for part in clients.values()) != len(indices):
        raise ValueError('O3 client partition does not cover training windows')
    return clients


def extra_client_part(corpus, fold, subject, scenario):
    mask = ((corpus.subjects == subject) & (corpus.scenarios == scenario)
            & corpus.valid)
    indices = np.flatnonzero(mask)
    if not len(indices):
        return None
    raw = corpus.x[indices].cuda()
    availability = corpus.availability[indices].cuda()
    encoded = ((features(raw, fold['summary']['feature_mode'])
                - fold['mean'].cuda()) / fold['std'].cuda())
    encoded = encoded * availability[:, None, :, None]
    return {'x': encoded, 'raw': raw, 'y': corpus.y[indices].cuda(),
            'a': availability}, indices


def evaluate_part(model, fold, part, batch_size, rotated=False, seed=0):
    view = {'data': {'target': part}, 'mean': fold['mean'], 'std': fold['std'],
            'summary': fold['summary']}
    return evaluate(model, view, 'target', batch_size,
                    rotated=rotated, rotation_seed=seed)


def source_metrics(model, fold, batch_size):
    metrics, _, _ = evaluate(model, fold, 'val_ideal', batch_size)
    return metrics['macro_f1']


def known_client_scores(model, corpus, fold, clients, batch_size,
                        local_states=None, local_keys=()):
    shared = clone_state(model.state_dict())
    report = {}
    for subject in clients:
        if local_states is not None:
            model.load_state_dict(apply_local_state(shared,
                local_states[subject], local_keys))
        subject_report = {}
        for scenario in ('self', 'induced'):
            extra = extra_client_part(corpus, fold, subject, scenario)
            if extra is None:
                continue
            part, indices = extra
            metrics, labels, predictions = evaluate_part(model, fold, part,
                                                         batch_size)
            subject_report[scenario] = {'metrics': metrics,
                'recordings': sorted(set(corpus.sources[indices].tolist())),
                'supported': metric_values(labels, predictions,
                                            np.unique(labels).tolist())}
        report[str(subject)] = subject_report
    model.load_state_dict(shared)
    return report


def evaluate_unseen(model, fold, batch_size):
    report = {}
    for scenario in ('ideal', 'self', 'induced'):
        if scenario not in fold['data']:
            continue
        metrics, labels, predictions = evaluate(model, fold, scenario, batch_size)
        report[scenario] = {'fixed33': metrics,
            'supported': metric_values(labels, predictions,
                                        np.unique(labels).tolist())}
    rotations = []
    for repeat in range(3):
        metric, _, _ = evaluate(model, fold, 'ideal', batch_size, rotated=True,
            rotation_seed=2026 + fold['summary']['test_subject'] * 101 + repeat)
        rotations.append(metric['macro_f1'])
    report['synthetic_rotated_ideal_f1'] = rotations
    return report


def train_centralized(corpus, fold, job, epochs, batch_size, output, smoke_limit):
    variant = Variant(job.kind, job.graph)
    model = FederatedSummerModel(job.kind, job.graph).cuda()
    optimizer = optimizer_for(model)
    scheduler = scheduler_for(optimizer, epochs) if smoke_limit is None else None
    checkpoint = output / 'latest.pt'
    best, best_epoch, first, history = -1.0, 0, 1, []
    if checkpoint.exists():
        state = torch.load(checkpoint, map_location='cpu', weights_only=False)
        model.load_state_dict(state['model'])
        optimizer.load_state_dict(state['optimizer'])
        if scheduler:
            scheduler.load_state_dict(state['scheduler'])
        best, best_epoch, first = state['best'], state['best_epoch'], state['epoch'] + 1
        history = state['history']
        restore_rng(state['rng'])
    with (output / 'epochs.jsonl').open('a', encoding='utf-8', buffering=1) as log:
        for epoch in range(first, epochs + 1):
            train = train_epoch(model, optimizer, fold, variant, batch_size,
                                limit=smoke_limit)
            score = source_metrics(model, fold, batch_size)
            row = {'epoch': epoch, 'train': train, 'source_validation_f1': score}
            history.append(row)
            log.write(json.dumps(row) + '\n')
            if score > best:
                best, best_epoch = score, epoch
                save_checkpoint(output / 'best.pt',
                    {'model': clone_state(model.state_dict()), 'epoch': epoch,
                     'source_validation_f1': score})
            if scheduler:
                scheduler.step()
            save_checkpoint(checkpoint, {'model': clone_state(model.state_dict()),
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict() if scheduler else None,
                'epoch': epoch, 'best': best, 'best_epoch': best_epoch,
                'history': history, 'rng': rng_state()})
            print(job.arm, job.method, job.subject, job.seed, epoch,
                  'source_val_F1', score, flush=True)
    model.load_state_dict(torch.load(output / 'best.pt', map_location='cpu',
                                      weights_only=False)['model'])
    clients = split_clients(corpus, fold)
    return {'best_epoch': best_epoch, 'best_validation_f1': best,
        'history': history, 'unseen': evaluate_unseen(model, fold, batch_size),
        'known_clients': known_client_scores(model, corpus, fold, clients,
                                             batch_size),
        'communication_bytes': 0}


def train_local_only(corpus, fold, job, epochs, batch_size, output, smoke_limit):
    variant = Variant(job.kind, job.graph)
    clients = split_clients(corpus, fold)
    seed_all(job.seed)
    initialized = clone_state(FederatedSummerModel(job.kind, job.graph).cuda().state_dict())
    report = {}
    for subject, part in clients.items():
        destination = output / 'clients' / str(subject)
        destination.mkdir(parents=True, exist_ok=True)
        model = FederatedSummerModel(job.kind, job.graph).cuda()
        model.load_state_dict(initialized)
        optimizer = optimizer_for(model)
        scheduler = scheduler_for(optimizer, epochs) if smoke_limit is None else None
        latest = destination / 'latest.pt'
        first = 1
        if latest.exists():
            state = torch.load(latest, map_location='cpu', weights_only=False)
            model.load_state_dict(state['model'])
            optimizer.load_state_dict(state['optimizer'])
            if scheduler:
                scheduler.load_state_dict(state['scheduler'])
            first = state['epoch'] + 1
            restore_rng(state['rng'])
        local_fold = dict(fold, data={'train': part})
        for epoch in range(first, epochs + 1):
            train_epoch(model, optimizer, local_fold, variant, batch_size,
                        limit=smoke_limit)
            if scheduler:
                scheduler.step()
            save_checkpoint(latest, {'model': clone_state(model.state_dict()),
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict() if scheduler else None,
                'epoch': epoch, 'rng': rng_state()})
        model.eval()
        report[str(subject)] = {}
        for scenario in ('self', 'induced'):
            extra = extra_client_part(corpus, fold, subject, scenario)
            if extra is None:
                continue
            part, _ = extra
            metric, labels, prediction = evaluate_part(model, fold, part,
                                                       batch_size)
            report[str(subject)][scenario] = {'fixed33': metric,
                'supported': metric_values(labels, prediction,
                                            np.unique(labels).tolist())}
        print(job.arm, job.method, job.subject, job.seed,
              'client', subject, 'complete', flush=True)
    return {'known_clients': report, 'unseen': None,
            'communication_bytes': 0,
            'selection': 'fixed final local epoch; no held-out placement used'}


def learning_rates(epochs):
    dummy = torch.nn.Linear(1, 1).cuda()
    optimizer = optimizer_for(dummy)
    scheduler = scheduler_for(optimizer, epochs)
    rates = []
    for _ in range(epochs):
        rates.append(optimizer.param_groups[0]['lr'])
        optimizer.zero_grad(set_to_none=True)
        dummy(torch.ones(1, 1, device='cuda')).sum().backward()
        optimizer.step()
        scheduler.step()
    return rates


def train_federated(corpus, fold, job, rounds, batch_size, output, smoke_limit):
    variant = Variant(job.kind, job.graph)
    clients = split_clients(corpus, fold)
    subjects = sorted(clients)
    counts = [len(clients[subject]['y']) for subject in subjects]
    model = FederatedSummerModel(job.kind, job.graph).cuda()
    local_keys = local_normalization_keys(model) if job.method == 'fedbn' else frozenset()
    server = clone_state(model.state_dict())
    client_norms = {subject: clone_state(server) for subject in subjects}
    rates = learning_rates(rounds) if smoke_limit is None else [1e-4] * rounds
    latest = output / 'latest.pt'
    first, best, best_round, history = 1, -1.0, 0, []
    if latest.exists():
        state = torch.load(latest, map_location='cpu', weights_only=False)
        server, client_norms = state['server'], state['client_norms']
        first, best, best_round = state['round'] + 1, state['best'], state['best_round']
        history = state['history']
        restore_rng(state['rng'])
    with (output / 'rounds.jsonl').open('a', encoding='utf-8', buffering=1) as log:
        for round_number in range(first, rounds + 1):
            started = time.perf_counter()
            states = []
            for subject in subjects:
                state = apply_local_state(server, client_norms[subject], local_keys)
                model.load_state_dict(state)
                optimizer = optimizer_for(model)
                optimizer.param_groups[0]['lr'] = rates[round_number - 1]
                local_fold = dict(fold, data={'train': clients[subject]})
                train_epoch(model, optimizer, local_fold, variant, batch_size,
                            limit=smoke_limit)
                updated = clone_state(model.state_dict())
                states.append(updated)
                if local_keys:
                    client_norms[subject] = updated
            server = weighted_state_average(states, counts, local_keys, server)
            reference = weighted_state_average(states, counts)
            evaluation_state = apply_local_state(server, reference, local_keys)
            model.load_state_dict(evaluation_state)
            score = source_metrics(model, fold, batch_size)
            bytes_this_round = round_communication_bytes(server, len(subjects),
                                                          local_keys)
            row = {'round': round_number, 'source_validation_f1': score,
                   'clients': subjects, 'counts': counts,
                   'communication_bytes': bytes_this_round,
                   'seconds': time.perf_counter() - started,
                   'lr': rates[round_number - 1]}
            history.append(row)
            log.write(json.dumps(row) + '\n')
            if score > best:
                best, best_round = score, round_number
                save_checkpoint(output / 'best.pt', {'server': server,
                    'reference': reference, 'client_norms': client_norms,
                    'round': round_number, 'source_validation_f1': score})
            save_checkpoint(latest, {'server': server,
                'client_norms': client_norms, 'round': round_number,
                'best': best, 'best_round': best_round,
                'history': history, 'rng': rng_state()})
            print(job.arm, job.method, job.subject, job.seed,
                  'round', round_number, 'source_val_F1', score, flush=True)
    best_state = torch.load(output / 'best.pt', map_location='cpu', weights_only=False)
    model.load_state_dict(apply_local_state(best_state['server'],
        best_state['reference'], local_keys))
    unseen = evaluate_unseen(model, fold, batch_size)
    known = known_client_scores(model, corpus, fold, clients, batch_size,
        best_state['client_norms'] if local_keys else None, local_keys)
    return {'best_round': best_round, 'best_validation_f1': best,
        'history': history, 'unseen': unseen, 'known_clients': known,
        'communication_bytes': sum(row['communication_bytes'] for row in history),
        'local_normalization_keys': sorted(local_keys),
        'source_reference_normalization': bool(local_keys)}


def run_job(corpus, fold, job, epochs, batch_size, output, smoke_limit=None):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    result_file = output / 'result.json'
    if result_file.exists():
        return json.loads(result_file.read_text())
    if job.method not in METHODS:
        raise ValueError(job.method)
    seed_all(job.seed)
    config = dict(asdict(job), epochs=epochs, batch_size=batch_size,
                  smoke_limit=smoke_limit,
                  recipe='summer Adam 1e-4/wd 5e-4, OneCycle per global pass',
                  client_policy='all source subjects, one local pass per round',
                  checkpoint_rule='source ideal macro-F1 only')
    config_file = output / 'config.json'
    if config_file.exists() and json.loads(config_file.read_text()) != config:
        raise ValueError('O3 job config changed')
    write_json(config_file, config)
    started = time.perf_counter()
    if job.method == 'centralized':
        result = train_centralized(corpus, fold, job, epochs, batch_size,
                                   output, smoke_limit)
    elif job.method == 'local_only':
        result = train_local_only(corpus, fold, job, epochs, batch_size,
                                  output, smoke_limit)
    else:
        result = train_federated(corpus, fold, job, epochs, batch_size,
                                 output, smoke_limit)
    result.update({'job': asdict(job), 'config': config,
                   'split': fold['summary'],
                   'seconds_this_invocation': time.perf_counter() - started})
    write_json(result_file, result)
    torch.cuda.empty_cache()
    return result
