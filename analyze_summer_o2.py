"""Paired, read-only analysis of summer-based O2 and frozen O1 controls."""

import argparse
import json
from pathlib import Path
import statistics


ROOT = Path(__file__).resolve().parent
O1 = ROOT / 'results' / 'summer_o1_session_20261017_232945' / 'training'
O1_CONTROLS = {
    'raw_physical': 'physical', 'raw_residual': 'residual',
    'raw_none': 'none', 'raw_aug_physical': 'physical_rotate',
    'raw_aug_residual': 'residual_rotate',
}


def analyze(root):
    root = Path(root).resolve()
    new = [json.loads(path.read_text()) for path in
           (root / 'jobs').glob('*/*/*/result.json')]
    if len(new) != 63:
        raise ValueError(f'Expected 63 new O2 results, found {len(new)}')
    rows = {(r['condition'], r['subject'], r['seed']): r for r in new}
    for label, original in O1_CONTROLS.items():
        for path in (O1 / 'jobs' / original).glob('*/*/result.json'):
            record = json.loads(path.read_text())
            rows[(label, record['subject'], record['seed'])] = record
    for label in [*O1_CONTROLS, *sorted({r['condition'] for r in new})]:
        count = sum(key[0] == label for key in rows)
        if count != 9:
            raise ValueError(f'{label} has {count}/9 matched records')
    labels = [*O1_CONTROLS, *sorted({r['condition'] for r in new})]
    summary = {}
    report = ['# Summer-based Objective 2 results', '',
        'Exploratory REALDISP study: held-out placement subjects 2, 5 and 15 '
        'had been inspected before this design. Scores are fixed 33-class '
        'macro-F1, averaged over seeds within subject and then over subjects.', '',
        '| Condition | Ideal | Self | Induced |', '|---|---:|---:|---:|']
    for label in labels:
        summary[label] = {}
        for scenario in ('ideal', 'self', 'induced'):
            by_subject = [statistics.mean(rows[(label, subject, seed)]
                ['metrics'][scenario]['macro_f1'] for seed in (13, 29, 47))
                for subject in (2, 5, 15)]
            summary[label][scenario] = {
                'mean': statistics.mean(by_subject),
                'subject_sd': statistics.stdev(by_subject),
                'by_subject': dict(zip((2, 5, 15), by_subject))}
        report.append('| ' + label + ' | ' + ' | '.join(
            f'{100 * summary[label][scenario]["mean"]:.2f}%'
            for scenario in ('ideal', 'self', 'induced')) + ' |')
    report += ['', '## Paired recorded-displacement differences', '']
    comparisons = []
    for kind in ('raw', 'raw_aug', 'magnitude', 'mag_dot'):
        dynamic, fixed, none = f'{kind}_residual', f'{kind}_physical', f'{kind}_none'
        for reference in (fixed, none):
            if dynamic not in labels or reference not in labels:
                continue
            differences = [rows[(dynamic, subject, seed)]['metrics']['induced']['macro_f1']
                - rows[(reference, subject, seed)]['metrics']['induced']['macro_f1']
                for subject in (2, 5, 15) for seed in (13, 29, 47)]
            comparisons.append({'target': dynamic, 'reference': reference,
                'mean_difference': statistics.mean(differences),
                'positive_pairs': sum(value > 0 for value in differences),
                'pairs': 9})
            report.append(f'- {dynamic} minus {reference}: '
                f'{100 * statistics.mean(differences):+.2f} points; '
                f'{sum(value > 0 for value in differences)}/9 positive pairs.')
    for graph in ('physical', 'residual', 'none'):
        for kind in ('magnitude', 'mag_dot'):
            target, reference = f'{kind}_{graph}', f'raw_{graph}'
            differences = [rows[(target, subject, seed)]['metrics']['induced']['macro_f1']
                - rows[(reference, subject, seed)]['metrics']['induced']['macro_f1']
                for subject in (2, 5, 15) for seed in (13, 29, 47)]
            comparisons.append({'target': target, 'reference': reference,
                'mean_difference': statistics.mean(differences),
                'positive_pairs': sum(value > 0 for value in differences),
                'pairs': 9})
            report.append(f'- {target} minus {reference}: '
                f'{100 * statistics.mean(differences):+.2f} points; '
                f'{sum(value > 0 for value in differences)}/9 positive pairs.')
    report += ['', '## Interpretation limits', '',
        'Inspect per-recording fixed and supported-class metrics, prediction '
        'files, synthetic-rotation scores, routing interventions and latency '
        'before selecting a model. Activity coverage differs by recording; '
        'subject, not overlapping windows, is the generalization unit. '
        'Static and no-graph controls remain controls. A feature gain alone '
        'does not establish useful dynamic routing. No independent recorded '
        'placement cohort was used for confirmation.']
    (root / 'ANALYSIS.md').write_text('\n'.join(report) + '\n', encoding='utf-8')
    (root / 'analysis_summary.json').write_text(json.dumps(
        {'conditions': summary, 'comparisons': comparisons}, indent=2),
        encoding='utf-8')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np
    figures = root / 'figures'
    figures.mkdir(exist_ok=True)
    figure, axis = plt.subplots(figsize=(12, 5), layout='constrained')
    for offset, scenario in ((-.26, 'ideal'), (0, 'self'), (.26, 'induced')):
        axis.bar(np.arange(len(labels)) + offset,
            [summary[label][scenario]['mean'] for label in labels], .25,
            label=scenario)
    axis.set(xticks=range(len(labels)), xticklabels=labels,
             ylabel='33-class macro-F1', title='Summer O2 placement comparison')
    axis.tick_params(axis='x', rotation=60)
    axis.legend()
    figure.savefig(figures / 'placement.png', dpi=180)
    plt.close(figure)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', required=True)
    analyze(Path(parser.parse_args().run))
