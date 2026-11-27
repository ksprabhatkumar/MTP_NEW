"""Read-only analysis of completed summer-based O1 results and scientific plots."""
import argparse
import json
from pathlib import Path
import statistics


def analyze(root):
    rows = [json.loads(p.read_text()) for p in (root/'jobs').glob('*/*/*/result.json')]
    names = sorted({r['condition'] for r in rows})
    report = ['# Summer-based Objective 1 results', '',
        f'Completed job records: {len(rows)}. Source: original summer repository commit '
        '`792ef355e8ba94da638c6b14b103aea0dabb0e07` with matched, documented O1 changes.', '',
        'Scores below average seeds within each subject, then subjects. Fixed 33-class macro-F1. '
        'The three subjects were inspected previously; this study is exploratory.', '',
        '| Condition | Jobs | Ideal F1 | Self F1 | Induced F1 |',
        '|---|---:|---:|---:|---:|']
    table = {}
    for name in names:
        selected=[r for r in rows if r['condition']==name]
        values={}
        for scenario in ('ideal','self','induced'):
            per_subject=[statistics.mean(r['metrics'][scenario]['macro_f1'] for r in selected if r['subject']==s)
                for s in (2,5,15) if any(r['subject']==s for r in selected)]
            values[scenario]={'mean':statistics.mean(per_subject),
                'subject_sd':statistics.stdev(per_subject) if len(per_subject)>1 else None}
        table[name]=values
        report.append(f'| {name} | {len(selected)}/9 | '+ ' | '.join(f'{100*values[s]["mean"]:.2f}%' for s in ('ideal','self','induced'))+' |')
    report += ['', '## Matched residual differences', '']
    for reference in ('static_sum','physical','none'):
        pairs=[]
        for r in rows:
            if r['condition']!='residual': continue
            other=next((q for q in rows if q['condition']==reference and q['subject']==r['subject'] and q['seed']==r['seed']),None)
            if other: pairs.append(r['metrics']['induced']['macro_f1']-other['metrics']['induced']['macro_f1'])
        if pairs:
            report.append(f'- Residual minus {reference}: {100*statistics.mean(pairs):+.2f} points across {len(pairs)} matched runs; positive in {sum(x>0 for x in pairs)}. Subjects, not overlapping windows, are generalization units.')
    report += ['', '## Interpretation and next checks', '',
        'A higher score alone is insufficient: inspect subject/seed variation, augmentation-only '
        'controls, ideal-placement cost and the recorded routing interventions in each result.json. '
        'The branch-balanced variant is an explicit alternative, not the authoritative raw mixture.', '',
        'All models use the restored summer GIN-attention and gated temporal pipeline except the '
        'external DynamicWHAR comparator. The principal loss is cross-entropy by user selection. '
        'GAE ablations add masked-target reconstruction. The six-axis common input, corrected '
        'data exclusions, training-only normalization and source-only checkpoint selection differ '
        'from historical summer results; do not claim numerical reproduction of the historical score.', '',
        'Exclude all-zero subject6_self, subject13_self and subject15_mutual4. Subject 15 therefore '
        'has induced recordings 5-7. Per-recording fixed/supported-class metrics and predictions '
        'are inside each job. No model choice for O2/O3 should be made solely from this aggregate table.']
    (root/'ANALYSIS.md').write_text('\n'.join(report)+'\n',encoding='utf-8')
    (root/'analysis_summary.json').write_text(json.dumps(table,indent=2),encoding='utf-8')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np
    figures=root/'figures'; figures.mkdir(exist_ok=True)
    fig,ax=plt.subplots(figsize=(13,6),layout='constrained')
    for offset,scenario in ((-.25,'ideal'),(0,'self'),(.25,'induced')):
        ax.bar(np.arange(len(names))+offset,[table[n][scenario]['mean'] for n in names],.24,
            yerr=[table[n][scenario]['subject_sd'] or 0 for n in names],label=scenario,capsize=2)
    ax.set(xticks=range(len(names)),xticklabels=names,ylabel='33-class macro-F1 (subject mean +/- SD)',
        title='Summer-backbone O1: matched placement comparison')
    ax.tick_params(axis='x',rotation=60); ax.legend()
    fig.savefig(figures/'placement.png',dpi=180); plt.close(fig)
    print(root/'ANALYSIS.md')


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run',required=True)
    analyze(Path(parser.parse_args().run))
