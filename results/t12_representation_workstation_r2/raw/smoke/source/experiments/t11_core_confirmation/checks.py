"""Workstation preflight, including fresh-process stochastic recovery."""
import argparse,copy,hashlib,json,subprocess,sys,time,traceback
from pathlib import Path
import torch
import numpy as np
from experiments.t10_temporal.model import TemporalFactorial,optimizer_groups
from experiments.t10_temporal.training import streams,make_teacher,loss_for_batch,update_teacher,corrupt_raw
from experiments.t11_core_confirmation.common import *
from experiments.t11_core_confirmation.data import build_fold
from run_o1_selection import rng_state,restore_rng
from workstation_experiments import seed_all
from summer_o1_experiments import scheduler_for
from o2_features import O2Corpus

def compare(a,b):
    if torch.is_tensor(a):torch.testing.assert_close(a.cpu(),b.cpu(),atol=1e-6,rtol=0)
    elif isinstance(a,np.ndarray):assert np.array_equal(a,b)
    elif isinstance(a,dict):
        assert a.keys()==b.keys()
        for k in a:compare(a[k],b[k])
    elif isinstance(a,(tuple,list)):
        assert len(a)==len(b)
        for x,y in zip(a,b):compare(x,y)
    else:assert a==b,(a,b)

def cpu_copy(x):
    if torch.is_tensor(x):return x.detach().cpu().clone()
    if isinstance(x,dict):return {k:cpu_copy(v) for k,v in x.items()}
    if isinstance(x,list):return [cpu_copy(v) for v in x]
    if isinstance(x,tuple):return tuple(cpu_copy(v) for v in x)
    return copy.deepcopy(x)

def objects(arm,temporal):
    model=TemporalFactorial(arm,temporal).cuda();teacher=make_teacher(model)
    opt=torch.optim.Adam(optimizer_groups(model),lr=1e-4,betas=(.9,.999),eps=1e-8,fused=False,foreach=False)
    return model,teacher,opt,scheduler_for(opt,31),streams(13)

def step(model,teacher,opt,sch,gens,batch,fold):
    model.train();teacher.eval();opt.zero_grad(set_to_none=True)
    loss,_,logits=loss_for_batch(model,teacher,batch,fold,gens['corruption'],3)
    assert torch.isfinite(loss) and logits.dtype==torch.float32 and torch.isfinite(logits).all()
    loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True)
    grads={n:float(p.grad.abs().sum()) for n,p in model.named_parameters() if p.grad is not None}
    opt.step();update_teacher(teacher,model);sch.step()
    return cpu_copy(dict(loss=loss,logits=logits,model=model.state_dict(),teacher=teacher.state_dict(),optimizer=opt.state_dict(),scheduler=sch.state_dict(),streams={k:g.get_state() for k,g in gens.items()},rng=rng_state())),grads

def worker(path):
    configure();p=Path(path);s=torch.load(p,map_location='cpu',weights_only=False)
    model,teacher,opt,sch,gens=objects(s['arm'],s['temporal'])
    for obj,key in ((model,'model'),(teacher,'teacher'),(opt,'optimizer'),(sch,'scheduler')):obj.load_state_dict(s[key])
    for k,g in gens.items():g.set_state(s['streams'][k])
    batch={k:v.cuda() for k,v in s['batch'].items()};restore_rng(s['rng'])
    result,_=step(model,teacher,opt,sch,gens,batch,s['fold'])
    atomic_torch(p.with_suffix('.actual.pt'),result)

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output');parser.add_argument('--worker');a=parser.parse_args()
    if a.worker:return worker(a.worker)
    assert a.output;out=Path(a.output).resolve();out.mkdir(parents=True,exist_ok=False);configure()
    with queue_lock(ROOT/'results/.t11_device.lock'),(out/'console.log').open('w',encoding='utf8') as log,console_log(log):
        started=time.perf_counter()
        try:
            fp=fingerprint();atomic_json(out/'environment.json',environment())
            # Official reference is bundled, unchanged, with its license/hashes.
            from experiments.t09_selective_ssm.check import reference,validation
            refdir=out/'reference_check';refdir.mkdir()
            manifest=json.loads((PACKAGE/'reference/reference.json').read_text())
            assert manifest['commit']=='e9594ce1c732d97440f0332fdc43170a2294dbfa'
            # A text-mode checkout may change line endings. Decode the exact
            # licensed upstream bytes and let the reference checker hash them.
            import base64
            cache=out/'reference_cache';cache.mkdir()
            for name,content in json.loads((PACKAGE/'reference_payload.json').read_text()).items():
                assert Path(name).name==name
                (cache/name).write_bytes(base64.b64decode(content,validate=True))
            reference_result=validation(reference(refdir,cache),refdir)
            atomic_json(out/'ssm_reference.json',reference_result)
            corpus=O2Corpus();initial={};records=[];fold_checks={}
            for subject in (2,5,15):
                fold=build_fold(corpus,subject);fold_checks[str(subject)]=fold['summary']
                train=set(fold['indices']['train']);assert not train.intersection(fold['indices']['val_ideal']) and not train.intersection(fold['indices']['val_self'])
                if subject==5:del fold;continue
                batch={k:v[:8].cuda() for k,v in fold['data']['train'].items()};batch['raw']=fold['raw_train'][:8].cuda()
                smallfold={k:fold[k] for k in ('mean','std','raw_scale')}
                for j in [j for j in selected('smoke') if j['subject']==subject]:
                    seed_all(13);model,teacher,opt,sch,gens=objects(j['arm'],j['temporal'])
                    key=(subject,j['temporal'],j['arm'])
                    initial[key]={k:v.cpu().clone() for k,v in model.state_dict().items() if not k.startswith('mamba.')}
                    model.eval();probe=batch['x'];av=batch['a']
                    with torch.no_grad():
                        _,_,diag=model(probe,av,True)
                        assert torch.count_nonzero(diag['correction'])==0
                        changed=probe.clone()
                        if j['arm']=='graph_repair':
                            changed[:,:,0]+=11;_,_,after=model(changed,av,True)
                            torch.testing.assert_close(diag['prediction'][:,:,0],after['prediction'][:,:,0],atol=1e-6,rtol=0)
                            assert (diag['prediction'][:,:,1:]-after['prediction'][:,:,1:]).abs().max()>0
                        if j['arm']=='local_repair':
                            changed[:,:,1:]+=11;_,_,after=model(changed,av,True)
                            torch.testing.assert_close(diag['prediction'][:,:,0],after['prediction'][:,:,0],atol=0,rtol=0)
                        missing=av.clone();missing[:,4]=False;changed=probe.clone();changed[:,:,4]=1e4
                        torch.testing.assert_close(model(changed,missing)[0],model(probe,missing)[0],atol=1e-6,rtol=0)
                        if j['arm']=='physical_robust':
                            b,t,n,_=probe.shape;x=probe*av[:,None,:,None]
                            adj=model.row_normalize(model.physical[None]*av[:,None,:])
                            expected=model.encoder(x.reshape(b*t,n,-1),adj[:,None].expand(-1,t,-1,-1).reshape(b*t,n,n)).reshape(b,t,n,-1)*av[:,None,:,None]
                            torch.testing.assert_close(expected,model.local_encode(probe,av),atol=0,rtol=0)
                    _,grads=step(model,teacher,opt,sch,gens,batch,smallfold)
                    if j['temporal']=='ssm':
                        for term in ('A_log','D','x_proj','dt_proj','conv1d'):assert any(term in k and v>0 for k,v in grads.items())
                    saved=cpu_copy(dict(arm=j['arm'],temporal=j['temporal'],model=model.state_dict(),teacher=teacher.state_dict(),optimizer=opt.state_dict(),scheduler=sch.state_dict(),streams={k:g.get_state() for k,g in gens.items()},rng=rng_state(),batch=batch,fold=smallfold))
                    path=out/(j['id']+'.pt');atomic_torch(path,saved)
                    expected,_=step(model,teacher,opt,sch,gens,batch,smallfold)
                    del model,teacher,opt,sch,gens;torch.cuda.empty_cache()
                    subprocess.run([sys.executable,'-m','experiments.t11_core_confirmation.checks','--worker',str(path)],cwd=ROOT,check=True,stdout=log,stderr=log)
                    actual=torch.load(path.with_suffix('.actual.pt'),map_location='cpu',weights_only=False);compare(expected,actual)
                    stress,mask=corrupt_raw(batch['raw'],batch['a'],smallfold['raw_scale'],streams(13)['corruption'])
                    h=hashlib.sha256(stress.cpu().numpy().tobytes()+mask.cpu().numpy().tobytes()).hexdigest()
                    records.append(dict(job=j['id'],subject=subject,passed=True,fresh_process_recovery=True,stress_sha256=h,gradient_norms=grads))
                    atomic_json(out/'partial_checks.json',records);print('PASS',j['id'],flush=True)
                del fold,batch
            for subject in (2,15):
                reference_initial=initial[(subject,'old','none_clean')]
                for (s,t,arm),state in initial.items():
                    if s==subject:
                        assert reference_initial.keys()==state.keys()
                        for k,v in reference_initial.items():torch.testing.assert_close(v,state[k],atol=0,rtol=0)
                assert len({r['stress_sha256'] for r in records if r['subject']==subject})==1
            assert fingerprint()==fp
            atomic_json(out/'preflight.json',dict(passed=True,cases=len(records),fingerprint=fp,records=records,folds=fold_checks,reference=reference_result,seconds=time.perf_counter()-started))
        except BaseException:
            error=traceback.format_exc();(out/'error.txt').write_text(error,encoding='utf8');atomic_json(out/'status.json',dict(state='failed',error=error));raise

if __name__=='__main__':main()
