"""Audit persisted predictions, checkpoint selection, controls and plot data."""
import json
from pathlib import Path
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from workstation_experiments import metric_values,write_json
ARMS=('none_frozen','local_repair_gated','graph_ungated','graph_repair_gated')
def report(out,results,curves,pairs):
    from experiments.t08_utility_gate_20261114.run import table
    assert len(results)==pairs*8
    for r in results:
        branch='local_repair' if r['arm'] in ('none_frozen','local_repair_gated') else 'graph_repair'
        job=out/'jobs'/branch/f'subject_{r["subject"]:02d}_seed_{r["seed"]}'
        with np.load(job/f'{r["arm"]}_{r["split"]}.npz') as p:
            assert np.isfinite(p['logits']).all()
            computed=metric_values(p['labels'],p['logits'].argmax(1))
            for k in ('accuracy','macro_f1','weighted_f1'):assert abs(computed[k]-r[k])<1e-10
            assert np.all(p['subjects']=={2:3,5:8}[r['subject']])
    orders={};stress={}
    for r in curves:
        key=r['subject'],r['seed'],r['phase'],r['epoch']
        assert key not in orders or orders[key]==r['order_hash'];orders[key]=r['order_hash']
        assert key not in stress or stress[key]==r['stress_hash'];stress[key]=r['stress_hash']
    for job in (out/'jobs').glob('*/*'):
        for phase in ('correction','gate'):
            latest=torch.load(job/f'{phase}_latest.pt',map_location='cpu',weights_only=False)
            best=torch.load(job/f'{phase}_best.pt',map_location='cpu',weights_only=False)
            assert latest['epoch']==latest['config']['epochs'][phase]
            assert best['epoch']==max(latest['history'],key=lambda x:x['ideal_f1'])['epoch']
            assert all(k in latest for k in ('optimizer','streams','cpu_rng','cuda_rng'))
    table(out/'metrics.csv',[{k:v for k,v in r.items() if k!='per_class'} for r in results])
    table(out/'per_class.csv',[dict(arm=r['arm'],subject=r['subject'],seed=r['seed'],split=r['split'],**c) for r in results for c in r['per_class']])
    table(out/'learning_curves.csv',curves)
    means={arm:{split:{k:float(np.mean([r[k] for r in results if r['arm']==arm and r['split']==split])) for k in ('accuracy','macro_f1')} for split in ('val_ideal','val_self')} for arm in ARMS}
    paired=[]
    for r in results:
        if r['arm']=='none_frozen':continue
        control=next(c for c in results if c['arm']=='none_frozen' and all(c[k]==r[k] for k in ('subject','seed','split')))
        paired.append(dict(arm=r['arm'],subject=r['subject'],seed=r['seed'],split=r['split'],**{k+'_delta_pp':100*(r[k]-control[k]) for k in ('accuracy','macro_f1')}))
    table(out/'paired_comparisons.csv',paired)
    write_json(out/'summary.json',dict(means=means,T08_complete=False,induced_evaluated=False,interpretation='Exploratory frozen-backbone pilot'))
    fig,axes=plt.subplots(1,2,figsize=(10,4))
    for branch in ('local_repair','graph_repair'):
        for ax,phase in zip(axes,('correction','gate')):
            r=[c for c in curves if c['branch']==branch and c['phase']==phase];epochs=sorted({c['epoch'] for c in r})
            ax.plot(epochs,[np.mean([c['ideal_f1'] for c in r if c['epoch']==e]) for e in epochs],label=branch);ax.set_title(phase+' ideal F1');ax.set_xlabel('epoch');ax.legend()
    fig.tight_layout();fig.savefig(out/'learning_curves.png',dpi=160);fig.savefig(out/'learning_curves.pdf');plt.close(fig)
    write_json(out/'audit.json',dict(passed=True,evaluated_arms=4,fold_seed_pairs=pairs,metrics_rows=len(results),checks=['saved prediction metrics','stage ideal-only checkpoint selection','matched ordering and corruption RNG','latest recovery state present'],T08_complete=False))
