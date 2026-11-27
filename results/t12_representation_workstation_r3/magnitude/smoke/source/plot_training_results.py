"""Export publication-ready PNG/PDF figures from saved JSON (no torch needed)."""
import argparse
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def main(root):
    root = Path(root)
    figure_dir = root / 'figures'
    figure_dir.mkdir(exist_ok=True)
    paths = list((root / 'jobs').glob('*/*/*/result.json'))
    if not paths:
        paths = list((root / 'smoke').glob('*/result.json'))
    results = [json.loads(path.read_text()) for path in paths]
    if not results:
        return
    fig, ax = plt.subplots(figsize=(9, 5), layout='constrained')
    for result in results:
        if result['subject'] == 2 and result['seed'] == 13:
            history = result['history']
            ax.plot([h['epoch'] for h in history], [h['validation_macro_f1'] for h in history],
                    label=result['condition'])
    ax.set(xlabel='Epoch', ylabel='Validation macro-F1 (33 classes)',
           title='Validation convergence: held-out subject 2, seed 13')
    ax.legend(fontsize=8, ncols=2)
    ax.grid(alpha=.2)
    fig.savefig(figure_dir / 'validation_convergence.png', dpi=180)
    fig.savefig(figure_dir / 'validation_convergence.pdf')
    plt.close(fig)
    selected = next((p for p in paths if json.loads(p.read_text())['condition'] == 'residual'), None)
    if selected:
        routing_path = selected.parent / 'routing.json'
        routing = json.loads(routing_path.read_text())
        fig, axes = plt.subplots(1, len(routing), figsize=(5*len(routing), 4), layout='constrained', squeeze=False)
        names = ['Back', 'LUA', 'RUA', 'LLA', 'RLA', 'LT', 'RT', 'LC', 'RC']
        for ax, (scenario, info) in zip(axes[0], routing.items()):
            im = ax.imshow(info['mean_adjacency'], vmin=0, vmax=1, cmap='Blues')
            ax.set(title=scenario, xlabel='Sender', ylabel='Receiver',
                   xticks=range(9), yticks=range(9), xticklabels=names, yticklabels=names)
            ax.tick_params(axis='x', rotation=90)
        fig.colorbar(im, ax=axes.ravel().tolist(), label='Mean routing weight')
        fig.suptitle('Residual graph: first 128 windows per scenario (descriptive, not paired)')
        fig.savefig(figure_dir / 'routing_comparison.png', dpi=180)
        fig.savefig(figure_dir / 'routing_comparison.pdf')
        plt.close(fig)
    # Uncertainty across subjects, after averaging seeds within each subject.
    primary = [r for r in results if r['subject'] in (2, 5, 15)]
    names = sorted(set(r['condition'] for r in primary))
    fig, ax = plt.subplots(figsize=(11, 5), layout='constrained')
    for offset, scenario in ((-.18, 'ideal'), (.18, 'induced')):
        means, errors = [], []
        for name in names:
            values = [np.mean([r['metrics'][scenario]['macro_f1'] for r in primary
                               if r['condition'] == name and r['subject'] == subject and scenario in r['metrics']])
                      for subject in (2, 5, 15) if any(r['condition'] == name and r['subject'] == subject and scenario in r['metrics'] for r in primary)]
            means.append(np.mean(values) if values else 0)
            errors.append(np.std(values, ddof=1) if len(values) > 1 else 0)
        ax.bar(np.arange(len(names)) + offset, means, .36, yerr=errors, capsize=3, label=scenario)
    ax.set(xticks=range(len(names)), xticklabels=names, ylabel='Macro-F1 (33 classes)',
           title='Primary placement comparison: subject mean and standard deviation')
    ax.tick_params(axis='x', rotation=35)
    ax.legend()
    fig.savefig(figure_dir / 'placement_comparison.png', dpi=180)
    fig.savefig(figure_dir / 'placement_comparison.pdf')
    plt.close(fig)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run_directory')
    main(parser.parse_args().run_directory)
