"""Source-validation-only causal audit of saved summer O1 routing checkpoints."""
import argparse
import json
import math
from pathlib import Path

import torch
import torch.nn.functional as F

from models.summer_dynamic import SummerDynamicWHAR
from o2_features import O2Corpus
from workstation_experiments import write_json


ROOT = Path(__file__).resolve().parent
O1 = ROOT / 'results' / 'summer_o1_session_20261017_232945' / 'training' / 'jobs'


def rms(tensor):
    return float(tensor.float().square().mean().sqrt())


def routed_forward(model, x, availability, adjacency):
    b, t, n, f = x.shape
    x = x * availability[:, None, :, None]
    mask = availability[:, None, :].expand(b, t, n).reshape(b*t, n, 1)
    latent = model.encoder(x.reshape(b*t, n, f), adjacency.reshape(b*t, n, n)) * mask
    temporal = model.mamba(latent.reshape(b, t, -1))
    temporal = temporal.reshape(b, t, n, model.hidden_dim) * availability[:, None, :, None]
    return model.classifier(model.norm_final(temporal.reshape(b, t, -1).mean(1)))


def mix(model, dynamic, availability):
    b, t, n, _ = dynamic.shape
    mask = availability[:, None, :].expand(b, t, n).reshape(b*t, n)
    off = 1-torch.eye(n, device=dynamic.device)
    sender = mask[:, None, :].float()*off
    physical = model.physical.expand(b*t, -1, -1)*sender
    global_graph = F.softplus(model.global_logits).expand(b*t, -1, -1)*sender
    components = torch.stack((physical, global_graph, dynamic.reshape(b*t, n, n)), 1)
    if model.graph_variant == 'residual_balanced':
        components = model.row_normalize(components)
    weights = torch.softmax(model.branch_logits, 0)
    if model.graph_variant in ('residual', 'residual_balanced'):
        weighted = components * weights[None, :, None, None]
        combined = weighted.sum(1)
    else:
        index = {'physical': 0, 'global': 1, 'dynamic': 2}.get(model.graph_variant)
        combined = torch.zeros_like(physical) if index is None else components[:, index]
        selected = torch.zeros(3, device=dynamic.device)
        if index is not None:
            selected[index] = 1
        weighted = components * selected[None, :, None, None]
    mass = weighted.sum(-1)
    effective = mass / mass.sum(1, keepdim=True).clamp_min(1e-12)
    return model.row_normalize(combined).reshape(b, t, n, n), effective.mean((0, 2))


def effect(model, x, a, y, baseline_a, baseline_logits, changed_a):
    changed_logits = routed_forward(model, x, a, changed_a)
    base_centered = baseline_logits - baseline_logits.mean(-1, keepdim=True)
    change_centered = changed_logits - changed_logits.mean(-1, keepdim=True)
    p = baseline_logits.softmax(-1).clamp_min(1e-12)
    q = changed_logits.softmax(-1).clamp_min(1e-12)
    messages = torch.matmul(baseline_a, x)
    changed_messages = torch.matmul(changed_a, x)
    return {'adjacency_rms_delta': rms(changed_a - baseline_a),
            'message_rms_delta': rms(changed_messages - messages),
            'centered_logit_rms_delta': rms(change_centered - base_centered),
            'probability_kl_mean': float((p*(p.log()-q.log())).sum(-1).mean()),
            'loss_delta': float(F.cross_entropy(changed_logits, y) - F.cross_entropy(baseline_logits, y)),
            'prediction_flip_fraction': float((changed_logits.argmax(-1) != baseline_logits.argmax(-1)).float().mean())}


def audit_one(fold, subject, graph, seed, sample_count):
    path = O1 / graph / f'subject_{subject:02d}' / f'seed_{seed}' / 'best.pt'
    state = torch.load(path, map_location='cpu', weights_only=False)
    model = SummerDynamicWHAR(graph_variant=graph).cuda().eval()
    model.load_state_dict(state['model'])
    part = fold['data']['val_ideal']
    eligible = torch.where(part['a'].all(-1))[0]
    if len(eligible) < 2:
        raise RuntimeError('Need at least two source-validation windows with all nine IMUs available')
    ix = eligible[torch.linspace(0, len(eligible)-1, min(sample_count, len(eligible)), device='cuda').long()]
    x, a, y = part['x'][ix], part['a'][ix], part['y'][ix]
    with torch.inference_mode():
        logits, _, diagnostics = model(x, a, return_diagnostics=True)
        baseline_a = diagnostics['adjacency'].float()
        recomputed_logits = routed_forward(model, x, a, baseline_a)
        forward_error = float((recomputed_logits-logits).abs().max())
        if forward_error > 2e-5:
            raise RuntimeError(f'Forward decomposition mismatch: {forward_error}')
        encoded = diagnostics['encoded'].float()
        scores = torch.bmm(model.q(encoded).float(), model.k(encoded).float().transpose(1, 2)) / math.sqrt(model.hidden_dim)
        raw_c = torch.softmax(scores, -1).reshape(len(x), x.shape[1], 9, 9)
        mask = (1-torch.eye(9, device=x.device))[None, None]
        dynamic = raw_c * mask
        mixed, mass = mix(model, dynamic, a)
        graph_error = float((mixed-baseline_a).abs().max())
        if graph_error > 2e-5:
            raise RuntimeError(f'Adjacency decomposition mismatch: {graph_error}')
        source_mean_c = dynamic.mean(0, keepdim=True).expand_as(dynamic)
        donor_c = dynamic.roll(1, 0)
        uniform_c = torch.full_like(dynamic, 1/9) * mask
        interventions = {}
        for name, c in [('uniform_affinity', uniform_c), ('source_mean_affinity', source_mean_c),
                        ('donor_affinity', donor_c)]:
            changed_a, _ = mix(model, c, a)
            interventions[name] = effect(model, x, a, y, baseline_a, logits, changed_a)
        for name, changed_a in [('source_mean_final_adjacency', baseline_a.mean(0, keepdim=True).expand_as(baseline_a)),
                                ('donor_final_adjacency', baseline_a.roll(1, 0))]:
            interventions[name] = effect(model, x, a, y, baseline_a, logits, changed_a)
        off = 1-torch.eye(9, device=x.device)
        forced_uniform = (off / 8)[None, None].expand_as(baseline_a)
        forced_physical = model.row_normalize(model.physical)[None, None].expand_as(baseline_a)
        for name, changed_a in [('forced_uniform_final_adjacency', forced_uniform),
                                ('forced_physical_final_adjacency', forced_physical)]:
            interventions[name] = effect(model, x, a, y, baseline_a, logits, changed_a)
        with torch.autocast('cuda', dtype=torch.bfloat16):
            bf16_logits, _, bf16_diag = model(x, a, return_diagnostics=True)
        report = {
            'checkpoint': str(path), 'subject': subject, 'seed': seed, 'graph': graph,
            'validation_subject': fold['summary']['validation_subject'],
            'sample_windows': len(x), 'selection': 'evenly spaced all-available source ideal validation windows',
            'forward_reconstruction_max_abs': forward_error,
            'adjacency_reconstruction_max_abs': graph_error,
            'raw_affinity_logit_std': float(scores.std()),
            'raw_affinity_logit_max_abs': float(scores.abs().max()),
            'encoded_descriptor_rms': rms(encoded),
            'query_weight_norm': float(model.q.weight.float().norm()),
            'key_weight_norm': float(model.k.weight.float().norm()),
            'raw_affinity_probability_std': float(raw_c.std()),
            'masked_dynamic_row_sum_mean': float(dynamic.sum(-1).mean()),
            'baseline_adjacency_between_window_std': float(baseline_a.std(0).mean()),
            'branch_weights': torch.softmax(model.branch_logits, 0).cpu().tolist(),
            'effective_branch_mass': mass.cpu().tolist(),
            'self_message_rms': rms((1+model.encoder.gin_agg.eps)*x),
            'neighbor_message_rms': rms(torch.matmul(baseline_a, x)),
            'fp32_bf16_logit_rms_delta': rms(bf16_logits.float()-logits),
            'fp32_bf16_adjacency_rms_delta': rms(bf16_diag['adjacency'].float()-baseline_a),
            'interventions': interventions,
        }
    del model, fold
    torch.cuda.empty_cache()
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True)
    parser.add_argument('--samples', type=int, default=96)
    parser.add_argument('--subjects', type=int, nargs='+', default=[2, 5])
    parser.add_argument('--seed', type=int, default=13)
    args = parser.parse_args()
    torch.set_num_threads(8)
    corpus = O2Corpus()
    rows = []
    for subject in args.subjects:
        fold = corpus.fold(subject, 'raw')
        for graph in ('residual', 'residual_balanced', 'physical', 'none'):
            print(f'Auditing source validation: held-out S{subject:02d}, {graph}, seed {args.seed}', flush=True)
            rows.append(audit_one(fold, subject, graph, args.seed, args.samples))
        del fold
        torch.cuda.empty_cache()
    report = {'source_only': True, 'rows': rows,
              'notes': 'Interventions compare consumed adjacency and centered logits; argmax alone is insufficient. '
                       'Source-validation windows only; held-out placement data did not enter this audit.'}
    write_json(Path(args.output), report)
    print(json.dumps({'output': args.output, 'audited_checkpoints': len(rows), 'passed': True}))


if __name__ == '__main__':
    main()
