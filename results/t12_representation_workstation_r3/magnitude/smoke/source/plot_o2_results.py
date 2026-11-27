"""Create source-validation and O2 placement figures from completed results."""
import json
import sys
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def main(directory):
    root = Path(directory)
    results = [json.loads(p.read_text()) for p in (root / 'jobs').glob('*/*/*/result.json')]
    if not results:
        return
    figures = root / 'figures'
    figures.mkdir(exist_ok=True)
    names = sorted({r['condition'] for r in results})
    fig, ax = plt.subplots(figsize=(11, 5), layout='constrained')
    for xoffset, scenario in ((-.25,'ideal'),(0,'self'),(.25,'induced')):
        means, errors = [], []
        for name in names:
            per_subject = [np.mean([r['metrics'][scenario]['macro_f1'] for r in results
                if r['condition']==name and r['subject']==subject and scenario in r['metrics']])
                for subject in (2,5,15) if any(r['condition']==name and r['subject']==subject
                    and scenario in r['metrics'] for r in results)]
            means.append(np.mean(per_subject) if per_subject else 0)
            errors.append(np.std(per_subject, ddof=1) if len(per_subject)>1 else 0)
        ax.bar(np.arange(len(names))+xoffset, means, .24, yerr=errors, capsize=2, label=scenario)
    ax.set(title='Objective 2 exploratory placement comparison',
        ylabel='33-class macro-F1; subject mean +/- SD', xticks=range(len(names)), xticklabels=names)
    ax.tick_params(axis='x', rotation=45)
    ax.legend()
    fig.savefig(figures/'placement_comparison.png',dpi=180)
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(11,5),layout='constrained')
    for result in results:
        if result['subject']==2 and result['seed']==13:
            ax.plot([row['epoch'] for row in result['history']],
                    [row['selection']['score'] for row in result['history']],label=result['condition'])
    ax.set(title='Source ideal/self validation selection: subject 2 test fold',
           xlabel='Epoch',ylabel='Selection score')
    ax.legend(ncols=2,fontsize=8)
    ax.grid(alpha=.25)
    fig.savefig(figures/'validation_convergence.png',dpi=180)
    plt.close(fig)


if __name__ == '__main__':
    main(sys.argv[1])
