"""Audit saved predictions and create reusable plot tables without test access."""
import argparse,csv,json
from pathlib import Path
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from workstation_experiments import metric_values,write_json
from models.summer_graph_repair import ARMS

def table(path,rows):
    with path.open('w',newline='',encoding='utf-8') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)

def analyze(output,expected):
    output=Path(output);paths=sorted(output.glob('jobs/*/*/result.json'));assert len(paths)==expected
    metrics=[];curves=[];orders={};records={}
    for path in paths:
        result=json.loads(path.read_text());cfg=result['config'];arm=cfg['arm'];subject=cfg['subject'];seed=cfg['seed']
        assert not result['split']['test_data_evaluated']
        latest=torch.load(path.parent/'latest.pt',map_location='cpu',weights_only=False)
        assert latest['epoch']==cfg['epochs'] and len(result['history'])==cfg['epochs']
        assert all(k in latest for k in ('teacher','optimizer','scheduler','streams','cpu_rng','cuda_rng'))
        best=torch.load(path.parent/'best.pt',map_location='cpu',weights_only=False)
        assert best['epoch']==result['best_epoch']
        best_expected=max(result['history'],key=lambda r:r['val_ideal']['macro_f1'])['epoch'];assert best_expected==result['best_epoch']
        records[arm,subject,seed]=result
        for split in ('val_ideal','val_self'):
            with np.load(path.parent/f'predictions_{split}.npz') as data:
                assert np.isfinite(data['logits']).all()
                calculated=metric_values(data['labels'],data['logits'].argmax(-1))
                assert np.all(data['subjects']==cfg['validation_subject'])
                for key in ('accuracy','macro_f1','weighted_f1'):assert abs(calculated[key]-result['metrics'][split][key])<1e-10
            metrics.append(dict(arm=arm,excluded_subject=subject,validation_subject=cfg['validation_subject'],seed=seed,split=split,best_epoch=result['best_epoch'],**{k:calculated[k] for k in ('accuracy','macro_f1','weighted_f1','windows')}))
        for row in result['history']:
            key=subject,seed,row['epoch'];digest=row['train']['order_sha256']
            assert key not in orders or orders[key]==digest;orders[key]=digest
            curves.append(dict(arm=arm,excluded_subject=subject,seed=seed,epoch=row['epoch'],**row['train'],ideal_accuracy=row['val_ideal']['accuracy'],self_accuracy=row['val_self']['accuracy'],ideal_f1=row['val_ideal']['macro_f1'],self_f1=row['val_self']['macro_f1'],lr=row['lr'],peak_cuda_allocated_mb=row['peak_cuda_allocated_mb']))
        del latest,best
    table(output/'metrics.csv',metrics);table(output/'learning_curves.csv',curves)
    means={arm:{split:{key:float(np.mean([r[key] for r in metrics if r['arm']==arm and r['split']==split])) for key in ('accuracy','macro_f1')} for split in ('val_ideal','val_self')} for arm in ARMS}
    deltas={control:{split:{key:100*(means['graph_repair'][split][key]-means[control][split][key]) for key in ('accuracy','macro_f1')} for split in ('val_ideal','val_self')} for control in ARMS if control!='graph_repair'}
    write_json(output/'summary.json',dict(means=means,graph_minus_control_percentage_points=deltas,completed_jobs=len(paths),test_evaluated=False,interpretation='One-seed source-validation pilot, not a held-out displacement claim'))
    fig,axes=plt.subplots(1,3,figsize=(15,4))
    for arm in ARMS:
        rows=[r for r in curves if r['arm']==arm];epochs=sorted({r['epoch'] for r in rows})
        for ax,key in zip(axes,('loss','ideal_accuracy','self_accuracy')):
            ax.plot(epochs,[np.mean([r[key] for r in rows if r['epoch']==e]) for e in epochs],label=arm);ax.set_xlabel('Epoch');ax.set_title(key.replace('_',' '));ax.grid(alpha=.2)
    axes[-1].legend(fontsize=7);fig.tight_layout();fig.savefig(output/'learning_curves.png',dpi=160);fig.savefig(output/'learning_curves.pdf');plt.close(fig)
    lines=['# Graph repair validation pilot','','Checkpoints selected only by source ideal macro-F1. Values are means over validation subjects; no held-out test evaluated.','','| Arm | Ideal accuracy | Self-placement accuracy | Self-placement macro-F1 |','|---|---:|---:|---:|']
    for arm in ARMS:
        m=means[arm];lines.append(f"| {arm} | {100*m['val_ideal']['accuracy']:.3f}% | {100*m['val_self']['accuracy']:.3f}% | {100*m['val_self']['macro_f1']:.3f}% |")
    lines+=['','Graph minus robust no-graph self-placement accuracy: '+f"{deltas['none_robust']['val_self']['accuracy']:+.3f} percentage points.",'','See metrics.csv for individual folds, learning_curves.csv for every epoch, and jobs/*/* for predictions, class metrics, checkpoints, and graph interventions.']
    (output/'REPORT.md').write_text('\n'.join(lines),encoding='utf-8')
    write_json(output/'audit.json',dict(passed=True,jobs=len(paths),checks=['predictions recomputed','best epoch selection verified','latest teacher/optimizer/RNG present','matched example ordering','no held-out test','plot tables exported']))
    print('\n'.join(lines),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',required=True);p.add_argument('--expected',type=int,required=True);a=p.parse_args();analyze(a.output,a.expected)
