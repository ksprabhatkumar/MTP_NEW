"""Supplementary CPU-only review of a finished pilot; never evaluates test data."""
import argparse,csv,hashlib,json
from pathlib import Path
import numpy as np
import torch
from sklearn.metrics import confusion_matrix
from models.summer_graph_repair import ARMS
from workstation_experiments import write_json
from analyze_graph_repair import table

def main():
    p=argparse.ArgumentParser();p.add_argument('--output',required=True);a=p.parse_args();root=Path(a.output)
    assert json.loads((root/'status.json').read_text())['state']=='complete'
    assert json.loads((root/'audit.json').read_text())['passed']
    jobs={};classes=[];graphs=[];confusions={};pairs=[]
    for path in sorted(root.glob('jobs/*/*/result.json')):
        r=json.loads(path.read_text());c=r['config'];key=(c['subject'],c['seed']);arm=c['arm'];jobs[arm,key]=(path.parent,r)
        for split,m in r['metrics'].items():
            for row in m['per_class']:classes.append(dict(arm=arm,excluded_subject=key[0],seed=key[1],split=split,**row))
            with np.load(path.parent/f'predictions_{split}.npz') as d:
                confusions[f'{arm}_subject{key[0]}_seed{key[1]}_{split}']=confusion_matrix(d['labels'],d['predictions'],labels=list(range(33)))
        with np.load(path.parent/'graph_samples.npz') as d:
            graph=dict(arm=arm,excluded_subject=key[0],seed=key[1],mean_null_mass=float(d['null_mass'].mean()),mean_quality=float(d['quality'].mean()),mean_gate=float(d['gate'].mean()),**r['diagnostics'])
            for mode,m in r['interventions'].items():
                graph[mode+'_accuracy']=m['accuracy'];graph[mode+'_macro_f1']=m['macro_f1']
            graphs.append(graph)
    keys=sorted({key for arm,key in jobs})
    for key in keys:
        graphdir,gr=jobs['graph_repair',key]
        for control in ('none_clean','none_robust','local_repair'):
            controldir,cr=jobs[control,key]
            for epoch in range(1,gr['config']['epochs']+1):
                gs=torch.load(graphdir/f'rng_epoch_{epoch:02d}.pt',weights_only=False)
                cs=torch.load(controldir/f'rng_epoch_{epoch:02d}.pt',weights_only=False)
                assert torch.equal(gs['order'],cs['order'])
                if control!='none_clean':assert torch.equal(gs['corruption'],cs['corruption'])
            for split in ('val_ideal','val_self'):
                with np.load(graphdir/f'predictions_{split}.npz') as g,np.load(controldir/f'predictions_{split}.npz') as c:
                    for field in ('labels','corpus_indices','subjects','recordings','scenarios','window_starts'):assert np.array_equal(g[field],c[field])
                    goodg=g['predictions']==g['labels'];goodc=c['predictions']==c['labels']
                    pairs.append(dict(control=control,excluded_subject=key[0],seed=key[1],split=split,windows=len(goodg),graph_only_correct=int((goodg&~goodc).sum()),control_only_correct=int((~goodg&goodc).sum()),both_correct=int((goodg&goodc).sum()),both_wrong=int((~goodg&~goodc).sum()),accuracy_difference_pp=100*(goodg.mean()-goodc.mean()),macro_f1_difference_pp=100*(gr['metrics'][split]['macro_f1']-cr['metrics'][split]['macro_f1'])))
    table(root/'per_class.csv',classes);table(root/'paired_comparisons.csv',pairs)
    fields=sorted({k for r in graphs for k in r});table(root/'graph_diagnostics.csv',[{k:r.get(k,'') for k in fields} for r in graphs])
    np.savez_compressed(root/'confusion_matrices.npz',**confusions)
    fingerprint=json.loads((root/'fingerprint.json').read_text());project=Path(__file__).resolve().parent
    assert all(hashlib.sha256((project/n).read_bytes()).hexdigest()==h for n,h in fingerprint['sources'].items())
    assert all(hashlib.sha256((project/'dataset/workstation_o1'/n).read_bytes()).hexdigest()==h for n,h in fingerprint['data'].items())
    write_json(root/'reproducibility_audit.json',dict(passed=True,jobs=len(jobs),checks=['source hashes unchanged','data hashes unchanged','all arms share original prediction row identities','all arms share order RNG state each epoch','robust arms share corruption RNG state each epoch','paired classification errors exported','per-class metrics and confusion matrices exported','graph reliability/null/intervention diagnostics exported'],limitations=['one seed and two source-validation subjects','overlapping windows are not independent replicates','no held-out induced performance claim']))
    print('Supplementary audit passed; paired results, confusion matrices, graph diagnostics and RNG alignment saved.')

if __name__=='__main__':main()
