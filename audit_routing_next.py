"""Inspect saved residual graphs and intervene on dynamic edges without retraining."""
import csv
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F

from workstation_experiments import Corpus, Condition, make_model

ROOT = Path(__file__).resolve().parent
RUN = ROOT / 'results' / 'o1_full_20261016_220301'
OUT = ROOT / 'results' / 'o2_preflight_20261017'


def uniform_dynamic_intervention(model):
    original = model.dynamic_graph.forward
    def replacement(x, physical_prior, availability=None, variant='residual'):
        adjacency, diagnostics = original(x, physical_prior, availability, variant)
        batch, nodes, _ = adjacency.shape
        if availability is None:
            senders = torch.ones(batch, nodes, device=x.device, dtype=torch.bool)
        else:
            senders = availability.to(device=x.device, dtype=torch.bool)
        mask = senders[:, None, :] & ~torch.eye(nodes, device=x.device, dtype=torch.bool)[None]
        uniform = mask.to(adjacency.dtype)
        uniform = uniform / uniform.sum(-1, keepdim=True).clamp_min(1)
        weights = diagnostics['branch_weights']
        mass = (weights[0] * diagnostics['physical'] +
                weights[1] * diagnostics['global'] + weights[2] * uniform) * mask
        adjacency = mass / mass.sum(-1, keepdim=True).clamp_min(1e-12)
        return adjacency, diagnostics
    model.dynamic_graph.forward = replacement
    return original


def main():
    OUT.mkdir(exist_ok=True)
    torch.set_num_threads(8)
    corpus = Corpus()
    rng = np.random.default_rng(17)
    rows = []
    for subject in (2, 5, 15):
        fold = corpus.fold(subject)
        for seed in (13, 29, 47):
            model = make_model(Condition('residual', 'residual'))
            checkpoint = torch.load(RUN / 'jobs' / 'residual' / f'subject_{subject:02d}' /
                                    f'seed_{seed}' / 'best.pt', map_location='cpu', weights_only=False)
            model.load_state_dict(checkpoint['model'])
            model.eval()
            weights = torch.softmax(model.dynamic_graph.branch_logits, 0).float().detach()
            row = {'subject': subject, 'seed': seed}
            for part in ('physical', 'global', 'dynamic'):
                row[f'logit_weight_{part}'] = float(weights[('physical', 'global', 'dynamic').index(part)])
            for scenario in ('ideal', 'self', 'induced'):
                x, y, a = fold['data'][scenario]
                indices = torch.as_tensor(rng.choice(len(y), min(384, len(y)), replace=False), device='cuda')
                adjacency_list, dynamic_list, mass_list = [], [], []
                changed, observed, logit_sum = 0, 0, 0.
                for take in indices.split(128):
                    bx, ba = x[take], a[take]
                    with torch.inference_mode(), torch.autocast('cuda', dtype=torch.bfloat16):
                        baseline, _, diagnostics = model(bx, ba, return_diagnostics=True)
                        original = uniform_dynamic_intervention(model)
                        try:
                            perturbed, _ = model(bx, ba)
                        finally:
                            model.dynamic_graph.forward = original
                    adjacency_list.append(diagnostics['adjacency'].float().cpu())
                    dynamic_list.append(diagnostics['dynamic'].float().cpu())
                    masses = torch.stack([
                        weights[0] * diagnostics['physical'].float().sum(-1),
                        weights[1] * diagnostics['global'].float().sum(-1),
                        weights[2] * diagnostics['dynamic'].float().sum(-1)], -1)
                    mass_list.append((masses / masses.sum(-1, keepdim=True).clamp_min(1e-12)).mean((0,1,2)).cpu())
                    changed += int((baseline.argmax(-1) != perturbed.argmax(-1)).sum())
                    observed += len(bx)
                    logit_sum += float((baseline.float() - perturbed.float()).abs().mean()) * len(bx)
                adj = torch.cat(adjacency_list)
                dynamic = torch.cat(dynamic_list)
                row[f'{scenario}_final_graph_std'] = float(adj.std((0, 1)).mean())
                row[f'{scenario}_dynamic_graph_std'] = float(dynamic.std((0, 1)).mean())
                row[f'{scenario}_intervention_prediction_flip_fraction'] = changed / observed
                row[f'{scenario}_intervention_mean_abs_logit_change'] = logit_sum / observed
                for name, value in zip(('physical', 'global', 'dynamic'), torch.stack(mass_list).mean(0)):
                    row[f'{scenario}_effective_mass_{name}'] = float(value)
            # Gradient diagnostic on source training data, without updating BN
            # running statistics or weights.
            model.zero_grad(set_to_none=True)
            tx, ty, ta = (v[:32] for v in fold['data']['train'])
            with torch.autocast('cuda', dtype=torch.bfloat16):
                logits, _ = model(tx, ta)
                loss = F.cross_entropy(logits, ty)
            loss.backward()
            row['q_grad_norm'] = float(model.dynamic_graph.q_proj.weight.grad.norm())
            row['global_grad_norm'] = float(model.dynamic_graph.global_logits.grad.norm())
            row['message_grad_norm'] = float(model.encoder.message_proj.weight.grad.norm())
            rows.append(row)
            print('Audited routing', subject, seed, 'induced flip',
                  row['induced_intervention_prediction_flip_fraction'], flush=True)
            del model
            torch.cuda.empty_cache()
        del fold
        torch.cuda.empty_cache()
    with (OUT / 'routing_intervention.csv').open('w', newline='', encoding='utf-8') as file:
        writer = csv.DictWriter(file, fieldnames=rows[0])
        writer.writeheader()
        writer.writerows(rows)


if __name__ == '__main__':
    main()
