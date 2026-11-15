"""Recompute source metrics from predictions, audit checkpoints, regenerate plots."""
import argparse
import csv
import json
from pathlib import Path
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from workstation_experiments import metric_values,write_json

def analyze(output,expected=None,epochs=None):
    output=Path(output)
    rows=[]; curves=[]; audits=[]
    jobs=sorted((output/'jobs').glob('*/*/result.json'))
    if expected is not None: assert len(jobs)==expected
    for path in jobs:
        result=json.loads(path.read_text()); config=result['config']; job=path.parent
        latest=torch.load(job/'latest.pt',map_location='cpu',weights_only=False)
        best=torch.load(job/'best.pt',map_location='cpu',weights_only=False)
        assert latest['epoch']==config['epochs'] and len(latest['history'])==config['epochs']
        if epochs is not None: assert latest['epoch']==epochs
        assert best['epoch']==result['best_epoch']
        for split in ('val_ideal','val_self'):
            with np.load(job/f'predictions_{split}.npz') as pred:
                assert len(pred['labels'])==len(pred['corpus_indices'])==len(pred['recordings'])
                assert np.array_equal(pred['logits'].argmax(-1),pred['predictions'])
                assert len(np.unique(pred['corpus_indices']))==len(pred['labels'])
                metrics=metric_values(pred['labels'],pred['predictions'])
                assert abs(metrics['macro_f1']-result['metrics'][split]['macro_f1'])<1e-12
            rows.append(dict(arm=config['arm'],subject=config['subject'],seed=config['seed'],
                split=split,macro_f1=metrics['macro_f1'],accuracy=metrics['accuracy'],best_epoch=best['epoch']))
        for epoch in result['history']:
            curves.append(dict(arm=config['arm'],subject=config['subject'],seed=config['seed'],
                epoch=epoch['epoch'],train_loss=epoch['train']['loss'],train_accuracy=epoch['train']['accuracy'],
                validation_loss=epoch['validation']['loss'],validation_f1=epoch['validation_ideal_f1'],
                validation_accuracy=epoch['validation']['accuracy'],lr=epoch['lr'],
                train_seconds=epoch['train']['seconds'],peak_cuda_allocated_mb=epoch['peak_cuda_allocated_mb']))
        audits.append(dict(job=str(job.relative_to(output)),epochs=latest['epoch'],passed=True))
    for name,values in [('metrics.csv',rows),('learning_curves.csv',curves)]:
        with (output/name).open('w',newline='') as f:
            writer=csv.DictWriter(f,fieldnames=list(values[0]));writer.writeheader();writer.writerows(values)
    summary={}
    for arm in sorted(set(r['arm'] for r in rows)):
        summary[arm]={split:float(np.mean([r['macro_f1'] for r in rows if r['arm']==arm and r['split']==split]))
                      for split in ('val_ideal','val_self')}
    paired=[]
    for row in rows:
        if row['arm']!='message_correction': continue
        for control in ('none','local_control'):
            ref=next(r for r in rows if r['arm']==control and all(r[k]==row[k] for k in ('subject','seed','split')))
            paired.append({**row,'control':control,'difference_pp':100*(row['macro_f1']-ref['macro_f1'])})
    write_json(output/'summary.json',dict(means=summary,paired=paired,source_only=True))
    fig,axes=plt.subplots(1,2,figsize=(11,4))
    for arm in summary:
        for index,key in enumerate(('train_loss','validation_f1')):
            selected=[r for r in curves if r['arm']==arm]
            xs=sorted(set(r['epoch'] for r in selected))
            ys=[np.mean([r[key] for r in selected if r['epoch']==e]) for e in xs]
            axes[index].plot(xs,ys,label=arm)
            axes[index].set(xlabel='Epoch',ylabel=key)
            axes[index].legend()
    fig.tight_layout();fig.savefig(output/'learning_curves.png',dpi=160);fig.savefig(output/'learning_curves.pdf');plt.close(fig)
    write_json(output/'completion_audit.json',dict(passed=True,jobs=audits,metrics_recomputed=True,plots_regenerated=True))
    print(json.dumps(summary,indent=2),flush=True)
    return summary

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('output');args=parser.parse_args();analyze(args.output)
