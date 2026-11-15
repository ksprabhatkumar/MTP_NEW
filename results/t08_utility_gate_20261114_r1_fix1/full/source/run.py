"""Bounded staged utility-gate pilot. No induced evaluation or hyperparameter search."""
import argparse,csv,hashlib,json,os,shutil,time,traceback
from pathlib import Path
import numpy as np
import torch
from sklearn.metrics import confusion_matrix
from experiments.t08_utility_gate_20261114.model import PredictiveRepair,load_backbone,set_stage,BACKBONE
from experiments.t08_utility_gate_20261114.training import loss_batch,streams
from experiments.t08_predictive_20261114.model import PredictiveRepair as Original
from experiments.t08_predictive_20261114.run import SOURCES
from o2_features import O2Corpus
from run_o1_selection import build_fold,rng_state,restore_rng
from run_local_message_pilot import precision_context
from workstation_experiments import write_json,save_checkpoint,seed_all,metric_values
ROOT=Path(__file__).resolve().parents[2]
SOURCE=ROOT/'results/t08_predictive_20261114_r1/full'

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def table(path,rows):
    if not rows:return
    with path.open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=sorted(set().union(*(r.keys() for r in rows))))
        writer.writeheader();writer.writerows(rows)

def frozen_state(model):return {k:v.detach().cpu().clone() for k,v in model.state_dict().items() if k.startswith(BACKBONE)}
def unchanged(model,reference):
    for k,v in model.state_dict().items():
        if k in reference:assert torch.equal(v.cpu(),reference[k]),'Frozen state changed: '+k

@torch.inference_mode()
def evaluate(model,part,path=None,metadata=None):
    model.eval();outputs=[];details={k:[] for k in ('p0','p1','pg','utility','quality','null_mass','gate')};norms=[];labels=[]
    for start in range(0,len(part['y']),64):
        b={k:v[start:start+64].cuda() for k,v in part.items() if k!='raw'}
        with precision_context():logits,_,d=model(b['x'],b['a'],True)
        assert torch.isfinite(logits).all()
        outputs.append(logits.float().cpu().numpy());labels.append(b['y'].cpu().numpy())
        if path is not None:
            for k in details:details[k].append(d[k].float().cpu().numpy())
            norms.append(d['correction'].float().square().mean((1,2,3)).sqrt().cpu().numpy())
    y=np.concatenate(labels);logits=np.concatenate(outputs);pred=logits.argmax(1);metrics=metric_values(y,pred)
    if path is not None:
        np.savez_compressed(path,labels=y,logits=logits,predictions=pred,confusion=confusion_matrix(y,pred,labels=list(range(33))),
            correction_rms=np.concatenate(norms),**{k:np.concatenate(v) for k,v in details.items()},**metadata)
    return metrics,logits

def train_stage(model,fold,phase,epochs,job,config):
    set_stage(model,phase);params=[p for p in model.parameters() if p.requires_grad]
    opt=torch.optim.Adam(params,lr=1e-4,weight_decay=5e-4)
    gens=streams(config['seed']+(0 if phase=='correction' else 50000))
    latest=job/(phase+'_latest.pt');bestpath=job/(phase+'_best.pt')
    history=[];first=1;best=-1.;best_epoch=0
    seed_all(config['seed']+(0 if phase=='correction' else 50000))
    if latest.exists():
        state=torch.load(latest,map_location='cpu',weights_only=False);assert state['config']==config and state['phase']==phase
        model.load_state_dict(state['model']);opt.load_state_dict(state['optimizer']);restore_rng(state)
        for k,g in gens.items():g.set_state(state['streams'][k])
        history=state['history'];first=state['epoch']+1;best=state['best'];best_epoch=state['best_epoch']
    reference={k:v.detach().cpu().clone() for k,v in model.state_dict().items() if phase=='gate' and not k.startswith('utility_gate.')}
    frozen=frozen_state(model)
    for epoch in range(first,epochs+1):
        set_stage(model,phase);data=fold['data']['train'];order=torch.randperm(len(data['y']),generator=gens['order']);total=0.;components={};started=time.perf_counter()
        stress_state=gens['corruption'].get_state().numpy().tobytes()
        for start in range(0,len(order),64):
            ix=order[start:start+64];batch={k:v[ix].cuda() for k,v in data.items()}
            opt.zero_grad(set_to_none=True);loss,parts=loss_batch(model,batch,fold,gens['corruption'],phase)
            loss.backward();torch.nn.utils.clip_grad_norm_(params,1.,error_if_nonfinite=True);opt.step()
            total+=float(loss.detach())*len(ix)
            for k,v in parts.items():components[k]=components.get(k,0.)+v*len(ix)
        unchanged(model,frozen);unchanged(model,reference)
        ideal,_=evaluate(model,fold['data']['val_ideal']);selfmetric,_=evaluate(model,fold['data']['val_self'])
        row=dict(phase=phase,epoch=epoch,loss=total/len(order),**{k:v/len(order) for k,v in components.items()},
            ideal_f1=ideal['macro_f1'],self_f1=selfmetric['macro_f1'],ideal_accuracy=ideal['accuracy'],self_accuracy=selfmetric['accuracy'],
            seconds=time.perf_counter()-started,order_hash=hashlib.sha256(order.numpy().tobytes()).hexdigest(),stress_hash=hashlib.sha256(stress_state).hexdigest())
        history.append(row)
        if ideal['macro_f1']>best:
            best=ideal['macro_f1'];best_epoch=epoch
            save_checkpoint(bestpath,dict(model=model.state_dict(),config=config,phase=phase,epoch=epoch,selection='ideal_macro_f1'))
        save_checkpoint(latest,dict(model=model.state_dict(),optimizer=opt.state_dict(),config=config,phase=phase,epoch=epoch,best=best,best_epoch=best_epoch,
            history=history,streams={k:g.get_state() for k,g in gens.items()},**rng_state()))
        table(job/(phase+'_history.csv'),history)
        write_json(job/'heartbeat.json',dict(phase=phase,epoch=epoch,pid=os.getpid(),time=time.time()))
        print(f'{config["branch"]} S{config["subject"]} seed{config["seed"]} {phase} {epoch}/{epochs} idealF1={best:.5f}',flush=True)
    state=torch.load(bestpath,map_location='cpu',weights_only=False);model.load_state_dict(state['model']);return history,state['epoch']

def initial_checks(model,original,fold,job):
    matmul_tf32=torch.backends.cuda.matmul.allow_tf32
    cudnn_tf32=torch.backends.cudnn.allow_tf32
    captures={};handles=[]
    for name,value in model.state_dict().items():
        if name.startswith(BACKBONE):
            assert torch.equal(value.cpu(),original.state_dict()[name].cpu()),'Backbone load mismatch: '+name
    def hook(label):
        def capture(module,inputs,output):
            captures.setdefault(label,[]).append(output.detach().float().cpu().clone())
        return capture
    for label,net in (('reference',original),('candidate',model)):
        for name in ('encoder','mamba','norm_final','classifier'):
            handles.append(getattr(net,name).register_forward_hook(hook(label+'.'+name)))
    try:
        torch.backends.cuda.matmul.allow_tf32=False
        torch.backends.cudnn.allow_tf32=False
        _initial_checks(model,original,fold,job)
    finally:
        details={}
        for name in ('encoder','mamba','norm_final','classifier'):
            refs=captures.get('reference.'+name,[]);cands=captures.get('candidate.'+name,[])
            if refs and cands:
                count=1 if name=='encoder' else 3
                details[name]=[float((v-refs[0]).abs().max()) for v in cands[:count] if v.shape==refs[0].shape]
        write_json(job/'equivalence_diagnostics.json',dict(precision='IEEE FP32',atol=1e-6,rtol=0,layer_max_abs=details))
        for handle in handles:handle.remove()
        torch.backends.cuda.matmul.allow_tf32=matmul_tf32
        torch.backends.cudnn.allow_tf32=cudnn_tf32

def _initial_checks(model,original,fold,job):
    b=fold['data']['val_ideal'];x=b['x'][:8].cuda();a=b['a'][:8].cuda()
    model.eval();original.eval()
    with torch.inference_mode():
        reference=original(x,a)[0];model.output_mode='ungated';out=model(x,a)[0]
        torch.testing.assert_close(out,reference,rtol=0,atol=1e-6)
        model.repair_output.weight.copy_(torch.eye(16,device='cuda'))
        model.force_gate=0.;model.output_mode='gated';out=model(x,a)[0]
        torch.testing.assert_close(out,reference,rtol=0,atol=1e-6)
        model.force_gate=None
        _,_,before=model(x,a,True);changed=x.clone();changed[:,:,0]+=100;_,_,after=model(changed,a,True)
        if model.arm=='graph_repair':
            torch.testing.assert_close(before['prediction'][:,:,0],after['prediction'][:,:,0],atol=1e-6,rtol=0)
            assert (before['prediction'][:,:,1:]-after['prediction'][:,:,1:]).abs().max()>1e-7
        else:torch.testing.assert_close(before['prediction'][:,:,1:],after['prediction'][:,:,1:],atol=1e-6,rtol=0)
        aa=a.clone();aa[:,4]=False;changed=x.clone();changed[:,:,4]=10000
        torch.testing.assert_close(model(x,aa)[0],model(changed,aa)[0],atol=1e-6,rtol=0)
        model.repair_output.weight.zero_();model.output_mode='ungated'
    write_json(job/'initial_checks.json',dict(passed=True,checks=['initial baseline equality','gate zero baseline fallback','receiver exclusion/local isolation','unavailable sensor invariance']))

def main():
    p=argparse.ArgumentParser();p.add_argument('--stage',choices=['smoke','full'],required=True);p.add_argument('--output',required=True);args=p.parse_args()
    assert torch.cuda.is_available();torch.set_num_threads(4);torch.backends.cudnn.benchmark=False;torch.backends.cuda.matmul.allow_tf32=True
    out=Path(args.output).resolve();out.mkdir(parents=True,exist_ok=True)
    import msvcrt
    with (out/'.lock').open('a+b') as lock:
        if lock.tell()==0:lock.write(b'0');lock.flush()
        lock.seek(0);msvcrt.locking(lock.fileno(),msvcrt.LK_NBLCK,1)
        try:
            start=time.time();sourcefp=json.loads((SOURCE/'fingerprint.json').read_text())
            assert all(sha(ROOT/k)==v for k,v in sourcefp['sources'].items())
            assert all(sha(ROOT/'dataset/workstation_o1'/k)==v for k,v in sourcefp['data'].items())
            assert json.loads((SOURCE/'audit.json').read_text())['passed']
            local_files=sorted(p for p in Path(__file__).parent.iterdir() if p.suffix in ('.py','.md','.ps1'))
            hashes={str(p.relative_to(ROOT)):sha(p) for p in local_files}
            if args.stage=='full':
                smoke=out.parent/'smoke'
                assert json.loads((smoke/'audit.json').read_text())['passed'],'Smoke must pass before full'
                smoke_fp=json.loads((smoke/'fingerprint.json').read_text())
                assert smoke_fp['sources']==hashes and smoke_fp['stage']=='smoke','Smoke source fingerprint mismatch'
                assert smoke_fp['upstream']==sourcefp,'Upstream changed after smoke'
                for branch in ('local_repair','graph_repair'):
                    assert json.loads((smoke/'jobs'/branch/'subject_02_seed_13'/'recovery.json').read_text())['passed']
            pairs=[(2,13)] if args.stage=='smoke' else [(s,k) for s in (2,5) for k in (13,29)]
            epochs={'correction':2,'gate':2} if args.stage=='smoke' else {'correction':8,'gate':4}
            fingerprint=dict(sources=hashes,upstream=sourcefp,pairs=pairs,epochs=epochs,stage=args.stage,
                checkpoints={f'{s}_{k}':sha(SOURCE/f'jobs/none_robust/subject_{s:02d}_seed_{k}/best.pt') for s,k in pairs})
            if (out/'fingerprint.json').exists():assert json.loads((out/'fingerprint.json').read_text())==json.loads(json.dumps(fingerprint)),'Resume fingerprint mismatch'
            write_json(out/'fingerprint.json',fingerprint)
            for path in local_files:
                target=out/'source'/path.name;target.parent.mkdir(exist_ok=True)
                if not target.exists():shutil.copy2(path,target)
            write_json(out/'status.json',dict(state='running',pid=os.getpid(),stage=args.stage))
            corpus=O2Corpus();results=[];curves=[]
            for subject,seed in pairs:
                fold=build_fold(corpus,subject);raw=corpus.x[fold['indices']['train']];fold['data']['train']['raw']=raw
                fold['raw_scale']=raw.std((0,1)).reshape(9,2,3).square().mean(-1).sqrt().clamp_min(1e-4).repeat_interleave(3,-1)
                save_checkpoint(out/f'normalization_{subject}.pt',dict(mean=fold['mean'],std=fold['std'],raw_scale=fold['raw_scale'],indices=fold['indices']))
                checkpoint=SOURCE/f'jobs/none_robust/subject_{subject:02d}_seed_{seed}/best.pt'
                state=torch.load(checkpoint,map_location='cpu',weights_only=False)
                torch.testing.assert_close(fold['mean'],state['mean'],atol=0,rtol=0);torch.testing.assert_close(fold['std'],state['std'],atol=0,rtol=0)
                original=Original('none_robust').cuda().eval();original.load_state_dict(state['model'])
                baseline_saved=False
                for branch in ('local_repair','graph_repair'):
                    job=out/'jobs'/branch/f'subject_{subject:02d}_seed_{seed}';job.mkdir(parents=True,exist_ok=True)
                    config=dict(subject=subject,seed=seed,branch=branch,epochs=epochs,backbone_sha=sha(checkpoint),validation_subject=fold['summary']['validation_subject'])
                    seed_all(seed+70000);model=PredictiveRepair(branch).cuda();load_backbone(model,state['model']);set_stage(model,'correction')
                    initial_checks(model,original,fold,job)
                    # Meaningful optimizer/RNG recovery check runs before smoke training only.
                    if args.stage=='smoke':
                        from experiments.t08_utility_gate_20261114.checks import recovery
                        recovery(model,fold,job,config)
                    for phase in ('correction','gate'):
                        history,best_epoch=train_stage(model,fold,phase,epochs[phase],job,config)
                        curves.extend(dict(branch=branch,subject=subject,seed=seed,**r) for r in history)
                        modes=['ungated'] if phase=='correction' else ['gated']
                        if not baseline_saved:modes+=['baseline']
                        for mode in modes:
                            if branch=='local_repair' and mode=='ungated':continue
                            arm='none_frozen' if mode=='baseline' else ('graph_ungated' if mode=='ungated' else branch+'_gated')
                            model.output_mode=mode
                            for split in ('val_ideal','val_self'):
                                ix=fold['indices'][split];meta=dict(corpus_indices=ix,subjects=corpus.subjects[ix],recordings=corpus.sources[ix],scenarios=corpus.scenarios[ix],window_starts=corpus.starts[ix].numpy())
                                metrics,logits=evaluate(model,fold['data'][split],job/f'{arm}_{split}.npz',meta)
                                if mode=='baseline':
                                    with np.load(checkpoint.parent/f'predictions_{split}.npz') as archived:
                                        assert np.array_equal(meta['corpus_indices'],archived['corpus_indices'])
                                        assert np.allclose(logits,archived['logits'],atol=1e-5,rtol=1e-5)
                                results.append(dict(arm=arm,subject=subject,seed=seed,split=split,best_epoch=0 if mode=='baseline' else best_epoch,**metrics))
                            if mode=='baseline':baseline_saved=True
                    del model;torch.cuda.empty_cache()
                del original,fold,raw;torch.cuda.empty_cache()
            assert all(sha(ROOT/k)==v for k,v in hashes.items())
            assert all(sha(ROOT/k)==v for k,v in sourcefp['sources'].items())
            assert all(sha(ROOT/'dataset/workstation_o1'/k)==v for k,v in sourcefp['data'].items())
            assert all(sha(SOURCE/f'jobs/none_robust/subject_{s:02d}_seed_{k}/best.pt')==fingerprint['checkpoints'][f'{s}_{k}'] for s,k in pairs)
            from experiments.t08_utility_gate_20261114.report import report
            report(out,results,curves,len(pairs))
            write_json(out/'status.json',dict(state='complete',pid=os.getpid(),stage=args.stage,seconds=time.time()-start))
        except BaseException:
            (out/'error.txt').write_text(traceback.format_exc());write_json(out/'status.json',dict(state='failed',pid=os.getpid()));raise
if __name__=='__main__':main()
