"""Independent fail-closed audit for saved T11 smoke/full predictions."""
import argparse,json
from pathlib import Path
import numpy as np
import torch
from workstation_experiments import metric_values
from experiments.t11_core_confirmation.common import selected
ROOT=Path(__file__).resolve().parents[2]
def audit(out,owner,stage):
 expected=selected(stage)
 records={};orders={};stresses={};metricrows=[]
 for j in expected:
  path=out/'jobs'/j['id'];r=json.loads((path/'result.json').read_text());cfg=r['config'];assert cfg['owner']==owner and cfg['stage']==stage and cfg['arm']==j['arm'] and cfg['temporal']==j['temporal'] and cfg['subject']==j['subject'] and cfg['seed']==j['seed'];assert r['split']['test_data_evaluated'] is False
  latest=torch.load(path/'latest.pt',map_location='cpu',weights_only=False);best=torch.load(path/'best.pt',map_location='cpu',weights_only=False)
  norm=torch.load(out/f"normalization_{j['subject']:02d}.pt",map_location='cpu',weights_only=False)
  assert latest['config']==cfg and best['config']==cfg
  for name in ('mean','std'):
   assert torch.equal(norm[name],latest[name]) and torch.equal(norm[name],best[name])
  assert j['subject'] not in r['split']['train_subjects'] and cfg['validation_subject'] not in r['split']['train_subjects']
  assert cfg['epochs']==(4 if stage=='smoke' else 31)
  assert latest['epoch']==cfg['epochs'] and len(r['history'])==cfg['epochs'] and best['epoch']==r['best_epoch'];assert all(k in latest for k in ('model','teacher','optimizer','scheduler','streams','cpu_rng','cuda_rng'))
  assert r['best_epoch']==max(r['history'],key=lambda x:x['val_ideal']['macro_f1'])['epoch']
  for split in ('val_ideal','val_self'):
   with np.load(path/f'predictions_{split}.npz') as d:
    y=d['labels'];logits=d['logits'];assert np.isfinite(logits).all() and logits.ndim==2 and logits.shape[1]==33 and len(y)==len(logits);m=metric_values(y,logits.argmax(-1))
    for k in ('accuracy','macro_f1','weighted_f1'):assert abs(m[k]-r['metrics'][split][k])<1e-10
    metricrows.append(dict(job=j['id'],arm=j['arm'],temporal=j['temporal'],subject=j['subject'],seed=j['seed'],split=split,**{k:float(m[k]) for k in ('accuracy','macro_f1','weighted_f1','windows')}))
    expectedval=cfg['validation_subject'];assert np.all(d['subjects']==expectedval);assert np.all(d['scenarios']==('ideal' if split=='val_ideal' else 'self'))
    assert np.array_equal(d['corpus_indices'],norm['indices'][split]) and not np.intersect1d(d['corpus_indices'],norm['indices']['train']).size
    assert np.array_equal(d['predictions'],logits.argmax(-1))
    from sklearn.metrics import confusion_matrix
    (path/f'confusion_{split}.json').write_text(json.dumps(confusion_matrix(y,logits.argmax(-1),labels=list(range(33))).tolist()))
  key=(j['subject'],j['seed']);oh=[x['train']['order_sha256'] for x in r['history']];assert len(oh)==cfg['epochs'];assert key not in orders or orders[key]==oh;orders[key]=oh
  if j['arm']!='none_clean':
   sh=[x['train']['stress_sha256'] for x in r['history']]
   assert key not in stresses or stresses[key]==sh,'actual corruption mismatch'
   stresses[key]=sh
  records[j['id']]=r
 assert len(records)==len(expected)
 import csv
 with (out/'metrics.csv').open('w',newline='',encoding='utf8') as f:
  w=csv.DictWriter(f,fieldnames=list(metricrows[0]));w.writeheader();w.writerows(metricrows)
 contrasts=[]
 by={(r['temporal'],r['subject'],r['seed'],r['arm'],r['split']):r for r in metricrows}
 for t in sorted({r['temporal'] for r in metricrows}):
  for subject,seed in sorted({(r['subject'],r['seed']) for r in metricrows}):
   for split in ('val_ideal','val_self'):
    for x,y in (('graph_repair','local_repair'),('graph_repair','none_robust')):
     left=by[(t,subject,seed,x,split)];right=by[(t,subject,seed,y,split)];label=x+'_minus_'+y
     contrasts.append(dict(temporal=t,subject=subject,seed=seed,split=split,contrast=label,accuracy_pp=100*(left['accuracy']-right['accuracy']),macro_f1_pp=100*(left['macro_f1']-right['macro_f1'])))
    if t=='ssm':
     for arm in ('none_clean','none_robust','physical_robust','local_repair','graph_repair'):
      left=by[('ssm',subject,seed,arm,split)];right=by[('old',subject,seed,arm,split)]
      contrasts.append(dict(temporal=t,subject=subject,seed=seed,split=split,contrast='ssm_minus_old:'+arm,accuracy_pp=100*(left['accuracy']-right['accuracy']),macro_f1_pp=100*(left['macro_f1']-right['macro_f1'])))
 atomic=(out/'paired_contrasts.json');atomic.write_text(json.dumps(contrasts,indent=2)+'\n')
 return dict(passed=True,owner=owner,stage=stage,expected_jobs=len(expected),audited_jobs=len(records),test_evaluated=False,order_stream_pairs=len(orders),recomputed_metrics=True,checkpoint_selection='ideal fixed33 macro-F1',paired_contrasts=len(contrasts))
def main():
 p=argparse.ArgumentParser();p.add_argument('--output',required=True);p.add_argument('--owner',required=True);p.add_argument('--stage',required=True);a=p.parse_args();out=Path(a.output);res=audit(out,a.owner,a.stage);(out/'audit.json').write_text(json.dumps(res,indent=2)+'\n');print(json.dumps(res))
if __name__=='__main__':main()
