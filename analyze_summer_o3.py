"""Read-only paired analysis of the two sequential O3 training groups."""

import argparse
import json
from pathlib import Path
import statistics


def load_group(path):
    return [json.loads(file.read_text()) for file in
            (Path(path) / 'jobs').glob('*/*/*/*/result.json')]


def known_self_values(row):
    values = []
    for client in row['known_clients'].values():
        score = client.get('self')
        if score is None:
            continue
        metrics = score.get('metrics', score.get('fixed33'))
        values.append(metrics['macro_f1'])
    return values


def analyze(baseline, localnorm, output):
    baseline, localnorm, output = map(Path, (baseline, localnorm, output))
    rows = load_group(baseline) + load_group(localnorm)
    if len(rows) != 144:
        raise ValueError(f'Expected 144 O3 result bundles, found {len(rows)}')
    indexed = {}
    for row in rows:
        job = row['job']
        key = (job['kind'], job['graph'], job['method'], job['subject'], job['seed'])
        if key in indexed:
            raise ValueError(f'Duplicate O3 job {key}')
        indexed[key] = row
    arms = sorted({(r['job']['kind'], r['job']['graph']) for r in rows})
    methods = ('centralized', 'local_only', 'fedavg', 'fedbn')
    report = ['# Summer-based Objective 3 results', '',
        'Exploratory subject-client simulator. Test and validation subjects '
        'were excluded from training clients. Unseen-user and known-client '
        'scores below are different evaluation tracks.', '',
        '| Arm | Method | Unseen induced F1 | Known self F1 | Worst known client F1 | Bytes/run |',
        '|---|---|---:|---:|---:|---:|']
    summary = {}
    for kind, graph in arms:
        arm = f'{kind}_{graph}'
        summary[arm] = {}
        for method in methods:
            selected = [indexed[(kind, graph, method, subject, seed)]
                for subject in (2, 5, 15) for seed in (13, 29, 47)]
            unseen = [row['unseen']['induced']['fixed33']['macro_f1']
                for row in selected if row.get('unseen')]
            known = [statistics.mean(known_self_values(row))
                for row in selected if known_self_values(row)]
            worst = [min(known_self_values(row))
                for row in selected if known_self_values(row)]
            bytes_mean = statistics.mean(row['communication_bytes'] for row in selected)
            item = {'runs': len(selected),
                'unseen_induced_mean': statistics.mean(unseen) if unseen else None,
                'known_self_mean': statistics.mean(known) if known else None,
                'worst_known_client_mean': statistics.mean(worst) if worst else None,
                'communication_bytes_per_run': bytes_mean}
            summary[arm][method] = item
            fmt = lambda value: f'{100*value:.2f}%' if value is not None else 'N/A'
            report.append(f'| {arm} | {method} | '
                f'{fmt(item["unseen_induced_mean"])} | '
                f'{fmt(item["known_self_mean"])} | '
                f'{fmt(item["worst_known_client_mean"])} | '
                f'{bytes_mean:,.0f} |')
    report += ['', '## Matched unseen-user differences', '']
    differences = []
    for kind, graph in arms:
        for method, reference in (('fedavg', 'centralized'),
                                  ('fedbn', 'fedavg')):
            values = [indexed[(kind, graph, method, subject, seed)]
                ['unseen']['induced']['fixed33']['macro_f1']
                - indexed[(kind, graph, reference, subject, seed)]
                ['unseen']['induced']['fixed33']['macro_f1']
                for subject in (2, 5, 15) for seed in (13, 29, 47)]
            comparison = {'arm': f'{kind}_{graph}', 'method': method,
                'reference': reference, 'mean_difference': statistics.mean(values),
                'positive_pairs': sum(value > 0 for value in values), 'pairs': 9}
            differences.append(comparison)
            report.append(f'- {kind}_{graph}, {method} minus {reference}: '
                f'{100*comparison["mean_difference"]:+.2f} points; '
                f'{comparison["positive_pairs"]}/9 positive pairs.')
    report += ['', '## Limits', '',
        'A simulator with one subject per client does not prove privacy, '
        'on-watch training or population-wide displacement generalization. '
        'Local-only has no unseen-user model. Compare client class support, '
        'recorded placements and synthetic rotation separately. The three '
        'induced subjects were previously examined, so these are exploratory '
        'results.']
    output.mkdir(parents=True, exist_ok=True)
    (output / 'ANALYSIS.md').write_text('\n'.join(report) + '\n', encoding='utf-8')
    (output / 'analysis_summary.json').write_text(json.dumps(
        {'arms': summary, 'comparisons': differences}, indent=2), encoding='utf-8')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np
    figures = output / 'figures'
    figures.mkdir(exist_ok=True)
    labels = [f'{kind}_{graph}' for kind, graph in arms]
    fig, ax = plt.subplots(figsize=(10, 5), layout='constrained')
    for offset, method in ((-.25, 'centralized'), (0, 'fedavg'), (.25, 'fedbn')):
        ax.bar(np.arange(len(labels)) + offset,
            [summary[arm][method]['unseen_induced_mean'] for arm in labels],
            .24, label=method)
    ax.set(xticks=range(len(labels)), xticklabels=labels,
           ylabel='Unseen-user induced 33-class macro-F1',
           title='Summer O3 federated placement comparison')
    ax.legend()
    fig.savefig(figures / 'unseen_placement.png', dpi=180)
    plt.close(fig)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--baseline', required=True)
    parser.add_argument('--localnorm', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    analyze(args.baseline, args.localnorm, args.output)
