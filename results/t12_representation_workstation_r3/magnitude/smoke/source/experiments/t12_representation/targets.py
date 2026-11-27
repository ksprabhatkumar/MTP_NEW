"""Frozen post-training target/rotation/dropout evaluation. Never used in smoke."""
import csv,json
import numpy as np
import torch
from sklearn.metrics import confusion_matrix
from o2_features import O2Corpus,features,random_rotations,rotate_raw
from workstation_experiments import metric_values
from experiments.t12_representation.model import TemporalFactorial
from .common import selected,atomic_json,sha,representation
from .data import evaluate

def evaluate_targets(out):
    corpus=O2Corpus();records=[];contrast_rows=[];target=out/'target_evaluation';target.mkdir(exist_ok=True)
    for job in selected('full'):
        source=out/'jobs'/job['id'];dest=target/job['id'];dest.mkdir(exist_ok=True)
        ck=torch.load(source/'best.pt',map_location='cpu',weights_only=False)
        assert ck['config']['subject']==job['subject'] and ck['config']['validation_subject']==job['validation_subject']
        model=TemporalFactorial(job['arm'],job['temporal']).cuda();model.load_state_dict(ck['model']);model.eval()
        jobrows=[]
        for scenario in ('ideal','self','induced'):
            ix=np.flatnonzero(corpus.valid & (corpus.subjects==job['subject']) & (corpus.scenarios==scenario))
            assert len(ix)>0,f'Missing target scenario {job}/{scenario}'
            raw=corpus.x[ix];av=corpus.availability[ix]
            metadata=dict(corpus_indices=ix,subjects=corpus.subjects[ix],recordings=corpus.sources[ix],scenarios=corpus.scenarios[ix],window_starts=corpus.starts[ix].numpy())
            def encode(x,availability):return dict(x=((features(x,representation())-ck['mean'])/ck['std'])*availability[:,None,:,None],a=availability,y=corpus.y[ix])
            part=encode(raw,av);artifact=dest/f'predictions_{scenario}.npz'
            metrics,y,pred=evaluate(model,part,64,artifact,metadata)
            with np.load(artifact) as d:
                check=metric_values(d['labels'],d['logits'].argmax(-1))
                for k in ('accuracy','macro_f1','weighted_f1'):assert abs(check[k]-metrics[k])<1e-10
                assert np.all(d['subjects']==job['subject']) and np.array_equal(d['corpus_indices'],ix)
            atomic_json(dest/f'{scenario}_metrics.json',dict(metrics=metrics,confusion=confusion_matrix(y,pred,labels=list(range(33))).tolist(),checkpoint_sha256=sha(source/'best.pt')))
            row=dict(job=job['id'],subject=job['subject'],seed=job['seed'],temporal=job['temporal'],arm=job['arm'],scenario=scenario,windows=len(y),**{k:metrics[k] for k in ('accuracy','macro_f1','weighted_f1')});records.append(row);jobrows.append(row)
            if scenario=='ideal':
                diagnostics=[]
                for seed in range(9001,9006):
                    gen=torch.Generator().manual_seed(seed);rot=rotate_raw(raw,random_rotations(len(raw),9,'cpu',gen))
                    changed=encode(rot,av);m,_,_=evaluate(model,changed,64,dest/f'rotation_{seed}.npz',metadata)
                    diagnostics.append(dict(kind='independent_fixed_rotation',seed=seed,metrics=m,feature_max_abs=float((changed['x']-part['x']).abs().max())))
                for sensor in range(9):
                    missing=av.clone();missing[:,sensor]=False
                    eligible=missing.any(-1);assert eligible.all(),'Unexpected all-missing sample after single dropout; report coverage before proceeding'
                    m,_,_=evaluate(model,encode(raw,missing),64,dest/f'dropout_{sensor}.npz',metadata)
                    diagnostics.append(dict(kind='single_sensor_dropout',sensor=sensor,metrics=m,coverage=float(eligible.float().mean())))
                atomic_json(dest/'stress_diagnostics.json',dict(records=diagnostics,all_missing_policy='abstain; never score as normal prediction'))
        atomic_json(dest/'summary.json',jobrows);del model;torch.cuda.empty_cache()
    assert len(records)==270
    with (target/'metrics.csv').open('w',newline='',encoding='utf8') as f:
        w=csv.DictWriter(f,fieldnames=list(records[0]));w.writeheader();w.writerows(records)
    by={(r['temporal'],r['subject'],r['seed'],r['arm'],r['scenario']):r for r in records}
    for subject in (2,5,15):
        for seed in (13,29,47):
            for scenario in ('ideal','self','induced'):
                for temporal in ('old','ssm'):
                    for control in ('none_robust','local_repair'):
                        a=by[(temporal,subject,seed,'graph_repair',scenario)];b=by[(temporal,subject,seed,control,scenario)]
                        contrast_rows.append(dict(subject=subject,seed=seed,scenario=scenario,temporal=temporal,contrast='graph_minus_'+control,macro_f1_pp=100*(a['macro_f1']-b['macro_f1']),accuracy_pp=100*(a['accuracy']-b['accuracy'])))
                for arm in ('none_clean','none_robust','physical_robust','local_repair','graph_repair'):
                    a=by[('ssm',subject,seed,arm,scenario)];b=by[('old',subject,seed,arm,scenario)]
                    contrast_rows.append(dict(subject=subject,seed=seed,scenario=scenario,arm=arm,contrast='ssm_minus_old',macro_f1_pp=100*(a['macro_f1']-b['macro_f1']),accuracy_pp=100*(a['accuracy']-b['accuracy'])))
    atomic_json(target/'paired_contrasts.json',contrast_rows)
    atomic_json(target/'audit.json',dict(passed=True,jobs=90,primary_rows=270,paired_contrasts=len(contrast_rows),test_used_for_selection=False,induced_previously_inspected=True,T08_complete=False))
