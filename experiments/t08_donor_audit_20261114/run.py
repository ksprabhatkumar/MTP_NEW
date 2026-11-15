"""Frozen-checkpoint audit; no optimizer, training or induced evaluation."""
import argparse,json,hashlib,traceback,time,csv
from pathlib import Path
import numpy as np
import torch
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import maximum_bipartite_matching
from sklearn.metrics import accuracy_score,f1_score,confusion_matrix
from o2_features import O2Corpus,features
from run_local_message_pilot import precision_context
from experiments.t08_predictive_20261114.model import PredictiveRepair as Original
from experiments.t08_donor_audit_20261114.model import PredictiveRepair
ROOT=Path(__file__).resolve().parents[2]
def save(p,x):p.write_text(json.dumps(x,indent=2))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def metrics(y,p):return dict(accuracy=float(accuracy_score(y,p)),macro_f1=float(f1_score(y,p,labels=list(range(33)),average='macro',zero_division=0)))
def mapping(y,recordings,mode,seed):
    rng=np.random.default_rng(seed);n=len(y);perm=rng.permutation(n)
    allowed=recordings[:,None]!=recordings[None,:]
    if mode=='same_activity':allowed &= y[:,None]==y[None,:]
    if mode=='different_activity':allowed &= y[:,None]!=y[None,:]
    # Randomized maximum matching: unique donors, no same-recording or self donor.
    matched=maximum_bipartite_matching(csr_matrix(allowed[:,perm]),perm_type='column')
    ix=np.flatnonzero(matched>=0);donor=perm[matched[ix]]
    assert len(np.unique(donor))==len(donor) and np.all(allowed[ix,donor])
    return ix,donor
@torch.inference_mode()
def infer(model,x,a,donors=None,zero=False):
    outputs=[];stats=[];model.eval()
    previous_intervention=model.message_intervention
    model.message_intervention='zero_correction' if zero else None
    try:
        for start in range(0,len(x),64):
            bx=x[start:start+64].cuda();ba=a[start:start+64].cuda();kw={}
            if donors is not None:kw=dict(donor_x=donors[0][start:start+64].cuda(),donor_availability=donors[1][start:start+64].cuda())
            with precision_context():logits,_,d=model(bx,ba,return_diagnostics=True,**kw)
            assert torch.isfinite(logits).all()
            outputs.append(logits.float().cpu().numpy())
            stats.append(np.stack([d[k].float().reshape(len(bx),-1).mean(1).cpu().numpy() for k in ('quality','gate','null_mass')]+[d[k].float().reshape(len(bx),-1).square().mean(1).sqrt().cpu().numpy() for k in ('prediction','correction')],1))
        return np.concatenate(outputs),np.concatenate(stats)
    finally:
        model.message_intervention=previous_intervention

def main():
    p=argparse.ArgumentParser();p.add_argument('--stage',choices=['smoke','full'],required=True);p.add_argument('--output',required=True);args=p.parse_args()
    out=Path(args.output);out.mkdir(parents=True,exist_ok=False)
    try:
        assert torch.cuda.is_available(),'CUDA required'
        torch.set_num_threads(4);torch.backends.cudnn.benchmark=False;torch.backends.cuda.matmul.allow_tf32=True
        source=ROOT/'results/t08_predictive_20261114_r1/full';fp=json.loads((source/'fingerprint.json').read_text())
        for name,h in fp['sources'].items():assert sha(ROOT/name)==h,name
        for name,h in fp['data'].items():assert sha(ROOT/'dataset/workstation_o1'/name)==h,name
        assert json.loads((source/'reproducibility_audit.json').read_text())['passed']
        manifest={str(p.relative_to(ROOT)):sha(p) for p in Path(__file__).parent.glob('*.py')}
        save(out/'manifest.json',dict(sources=manifest,stage=args.stage,device=torch.cuda.get_device_name(),torch=torch.__version__,started=time.time()))
        corpus=O2Corpus();rows=[];jobs=[(2,13)] if args.stage=='smoke' else [(s,k) for s in (2,5) for k in (13,29)]
        for subject,seed in jobs:
            job=source/'jobs/graph_repair'/f'subject_{subject:02d}_seed_{seed}'
            checkpoint=job/'best.pt';state=torch.load(checkpoint,map_location='cpu',weights_only=False)
            model=PredictiveRepair('graph_repair').cuda().eval();model.load_state_dict(state['model'],strict=True)
            original=Original('graph_repair').cuda().eval();original.load_state_dict(state['model'],strict=True)
            save(out/f'checkpoint_{subject}_{seed}.json',dict(path=str(checkpoint),sha256=sha(checkpoint),epoch=state['epoch']))
            for split in ('val_ideal','val_self'):
                with np.load(job/f'predictions_{split}.npz') as z:archive={k:z[k].copy() for k in z.files}
                ix=archive['corpus_indices'];y=archive['labels'];a=corpus.availability[ix]
                assert np.array_equal(corpus.y[ix].numpy(),y)
                x=torch.cat([features(corpus.x[ix[i:i+256]],'mag_dot') for i in range(0,len(ix),256)])
                x=(x-state['mean'])/state['std'];x*=a[:,None,:,None]
                assert model.message_intervention is None,'Intervention state leaked across splits'
                assert original.message_intervention is None
                with torch.inference_mode(),precision_context():
                    b=x[:64].cuda();av=a[:64].cuda();ref=original(b,av)[0];got=model(b,av)[0]
                    assert torch.allclose(ref,got,atol=1e-6,rtol=1e-6),'Model-copy equivalence failed'
                # Exercise the exact sequence that previously contaminated the next split.
                infer(model,x[:64],a[:64],zero=True)
                assert model.message_intervention is None,'Zero-correction state was not restored'
                with torch.inference_mode(),precision_context():
                    restored=model(b,av)[0]
                    assert torch.allclose(ref,restored,atol=1e-6,rtol=1e-6),'Normal output changed after zero-correction test'
                    explicit=model(b,av,donor_x=b,donor_availability=av)[0]
                    assert torch.allclose(ref,explicit,atol=1e-6,rtol=1e-6),'Explicit identity donors differ'
                save(out/f'equivalence_{subject}_{seed}_{split}.json',dict(passed=True,
                    normal_max_abs=float((ref-got).abs().max()),restored_max_abs=float((ref-restored).abs().max()),
                    identity_donor_max_abs=float((ref-explicit).abs().max()),atol=1e-6,rtol=1e-6))
                normal,diagnostics=infer(model,x,a)
                assert np.array_equal(normal.argmax(1),archive['predictions']),'Archived labels disagree'
                assert np.allclose(normal,archive['logits'],atol=1e-5,rtol=1e-5),'Archived logits disagree'
                prefix=f's{subject}_seed{seed}_{split}'
                np.savez_compressed(out/(prefix+'_normal.npz'),logits=normal,diagnostics=diagnostics,**{k:v for k,v in archive.items() if k!='logits'})
                conditions=[('zero_correction',0,np.arange(len(y)),np.arange(len(y)))]
                for mode in ('global','same_activity','different_activity'):
                    for ps in ([101] if args.stage=='smoke' else range(101,111)):
                        ri,di=mapping(y,archive['recordings'],mode,ps);conditions.append((mode,ps,ri,di))
                for mode,ps,ri,di in conditions:
                    common=dict(subject=subject,validation_subject=state['config']['validation_subject'],seed=seed,split=split,mode=mode,permutation_seed=ps,eligible=len(ri),total=len(y))
                    if not len(ri):rows.append(dict(**common,status='INFEASIBLE'));continue
                    logits,stats=infer(model,x[ri],a[ri],None if mode=='zero_correction' else (x[di],a[di]),zero=mode=='zero_correction')
                    m=metrics(y[ri],logits.argmax(1));base=metrics(y[ri],normal[ri].argmax(1))
                    rows.append(dict(**common,status='OK',**m,**{k+'_difference_pp':100*(m[k]-base[k]) for k in m}))
                    np.savez_compressed(out/f'{prefix}_{mode}_{ps}.npz',receiver_indices=ix[ri],donor_indices=ix[di],labels=y[ri],logits=logits,normal_logits=normal[ri],diagnostics=stats,receiver_recordings=archive['recordings'][ri],donor_recordings=archive['recordings'][di],receiver_starts=archive['window_starts'][ri],donor_starts=archive['window_starts'][di],confusion=confusion_matrix(y[ri],logits.argmax(1),labels=list(range(33))))
                    save(out/'heartbeat.json',dict(**common,time=time.time()))
                print(prefix+' complete',flush=True)
            assert sha(checkpoint)==json.loads((out/f'checkpoint_{subject}_{seed}.json').read_text())['sha256']
            del original,model;torch.cuda.empty_cache()
        fields=sorted(set().union(*(r.keys() for r in rows)))
        with (out/'metrics.csv').open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(rows)
        for name,h in manifest.items():assert sha(ROOT/name)==h
        save(out/'audit.json',dict(passed=True,stage=args.stage,jobs=len(jobs),rows=len(rows),infeasible=sum(r['status']=='INFEASIBLE' for r in rows),training_performed=False,induced_evaluated=False,notes='Diagnostic stats columns: quality,gate,null_mass,prediction_rms,correction_rms; one row per receiver. Review per subject; no automatic promotion.'))
    except BaseException:
        (out/'error.txt').write_text(traceback.format_exc());raise
if __name__=='__main__':main()
