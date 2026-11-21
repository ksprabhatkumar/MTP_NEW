"""Descriptive source-only selection decision from the frozen completed matrix."""
import json
import csv
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def main():
    import argparse
    parser=argparse.ArgumentParser()
    parser.add_argument('--output',default='results/o1_t01_t08_20261028/full')
    parser.add_argument('--graph-arm',default='gated_neighbor')
    args=parser.parse_args()
    root=Path(__file__).resolve().parent
    out=root/args.output
    assert json.loads((out/'completion_audit.json').read_text())['passed']
    summary=json.loads((out/'summary.json').read_text())
    graph_arm=args.graph_arm
    means=summary['means']; g=means[graph_arm]; n=means['none']; l=means['local_control']
    # In this source pilot, config.subject is excluded from training, NOT the
    # subject whose validation predictions are evaluated. Make that explicit.
    with (out/'metrics.csv').open(newline='') as handle:
        contextual=list(csv.DictReader(handle))
    for row in contextual:
        fold=json.loads((out/f"fold_{int(row['subject']):02d}.json").read_text())
        row['excluded_test_subject']=row.pop('subject')
        row['evaluation_subject']=fold['validation_subject']
        row['evaluation_role']='source_validation'
    with (out/'plot_metrics.csv').open('w',newline='') as handle:
        writer=csv.DictWriter(handle,fieldnames=list(contextual[0]));writer.writeheader();writer.writerows(contextual)
    effects={split:{'vs_none_pp':100*(g[split]-n[split]),'vs_local_pp':100*(g[split]-l[split])}
             for split in ('val_ideal','val_self')}
    passed=g['val_self']>n['val_self'] and g['val_self']>=l['val_self'] and g['val_ideal']>=n['val_ideal']-.005
    interventions=[]
    for p in sorted((out/'jobs'/graph_arm).glob('*/result.json')):
        r=json.loads(p.read_text()); checks=json.loads((p.parent/'interventions.json').read_text())
        for split in ('val_ideal','val_self'):
            interventions.append(dict(subject=r['config']['subject'],seed=r['config']['seed'],split=split,
                **{f'{mode}_minus_normal_pp':100*(m['macro_f1']-r['metrics'][split]['macro_f1']) for mode,m in checks[split].items()}))
    result=dict(candidate=graph_arm,promotion_criteria_passed=passed,
        state='candidate_selected' if passed else 'provisional_candidate_not_promoted',
        means=means,effects=effects,paired=summary['paired'],interventions=interventions,
        graph_superiority_on_displacement='NOT EVALUATED',
        scope='Two source-validation folds, two seeds, sixteen epochs; not independent displacement confirmation.')
    (out/'selection_decision.json').write_text(json.dumps(result,indent=2))
    fig,axes=plt.subplots(1,2,figsize=(10,4),sharey=True)
    for ax,split in zip(axes,('val_ideal','val_self')):
        for x,control in enumerate(('none','local_control')):
            points=[r['difference_pp'] for r in summary['paired'] if r['split']==split and r['control']==control]
            ax.scatter(np.full(len(points),x)+np.linspace(-.07,.07,len(points)),points)
            ax.plot([x-.2,x+.2],[np.mean(points)]*2,color='black')
        ax.axhline(0,color='grey',linestyle='--');ax.set_xticks([0,1],['vs no graph','vs local control']);ax.set_title(split)
        ax.set_ylabel('Graph minus control macro-F1 (percentage points)')
    fig.suptitle('Source pilot: four paired fold/seed effects; black line = mean')
    fig.tight_layout();fig.savefig(out/'paired_effects.png',dpi=160);fig.savefig(out/'paired_effects.pdf');plt.close(fig)
    graph_jobs=sorted((out/'jobs'/graph_arm).glob('*/graph_samples.npz'))
    if graph_jobs:
        fig,axes=plt.subplots(2,len(graph_jobs),figsize=(4*len(graph_jobs),7),squeeze=False)
        for column,path in enumerate(graph_jobs):
            with np.load(path) as samples:
                adjacency=samples['adjacency'].mean((0,1))
                gates=samples['gate'].reshape(-1,9).mean(0)
            axes[0,column].imshow(adjacency,vmin=0,vmax=1,cmap='viridis')
            axes[0,column].set(title=path.parent.name,xlabel='Sender sensor index',ylabel='Receiver sensor index')
            axes[1,column].bar(np.arange(9),gates)
            gate_label='Fixed external multiplier' if graph_arm=='message_correction' else 'Mean receiver gate'
            axes[1,column].set(ylim=(0,1.05),xlabel='Sensor index',ylabel=gate_label)
        fig.suptitle('Fixed source-validation samples; these are not dataset-wide averages')
        fig.tight_layout();fig.savefig(out/'graph_diagnostics.png',dpi=160);fig.savefig(out/'graph_diagnostics.pdf');plt.close(fig)
    lines=['# T08 source-only candidate decision','',f'Promotion criteria passed: {passed}.',
           f'Candidate: {graph_arm}. '+('Proceed to matched workstation confirmation.' if passed else 'Do not promote as a validated improvement; retain as a provisional mechanism experiment for workstation review.'),'',
           'No recorded-displacement test was evaluated. O1 superiority remains unresolved.','',
           '| Arm | Source ideal F1 % | Source self F1 % |','|---|---:|---:|']
    for arm,m in means.items(): lines.append(f"| {arm} | {100*m['val_ideal']:.4f} | {100*m['val_self']:.4f} |")
    lines += ['', 'All twelve jobs, checkpoint recovery, prediction metrics and plot regeneration are audited. Selection is based on the frozen source criteria, not a test score. Four fold/seed pairs are not four independent subjects. Compare graph interventions in selection_decision.json before attributing any benefit to neighbor information. Historical eight-epoch corrections are not a matched comparator to this sixteen-epoch experiment.']
    donor_path=out/'distant_donor_audit.json'
    if donor_path.exists():
        donor=json.loads(donor_path.read_text())
        effects={split:float(np.mean([r['difference_pp'] for r in donor['rows'] if r['split']==split])) for split in ('val_ideal','val_self')}
        result['distant_donor_mean_effect_pp']=effects
        result['interpretation']='Neighbor values are useful versus distant donors on average, but the trained graph fails the matched source-self promotion criterion. No displacement superiority established.'
        lines += ['',f'Distant-window donor mean effects (percentage points): {effects}. Receiver local features and adjacency were retained; donors were offset by half the complete validation set. This demonstrates some useful message dependence, not superiority over a separately trained no-graph model.',
                  '', 'Next research decision: test whether small initial gates slow neighbor learning, or whether matched source-side sensor perturbations are needed to teach rejection. Freeze that experiment before training; preserve identical perturbation and optimization exposure in controls.',
                  '', 'Gate initialization is an optimization variable: delta=g W_o phi(m), with W_o=0. Under plain SGD the first effective correction update scales approximately as g squared; Adam changes that scaling through moment normalization. Consult config.json for this run\'s gate bias and the frozen experiment specification before proposing another ablation.']
        (out/'selection_decision.json').write_text(json.dumps(result,indent=2))
    (out/'T08_DECISION.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps(result,indent=2))

if __name__=='__main__': main()
