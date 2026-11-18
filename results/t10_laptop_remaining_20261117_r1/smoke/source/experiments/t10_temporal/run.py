"""Portable, fail-closed T10 source-validation runner for Linux Colab and Windows."""
import argparse, contextlib, hashlib, json, os, random, shutil, sys, time, traceback
from datetime import datetime
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
from experiments.t10_temporal.model import ARMS, TEMPORALS, TemporalFactorial, optimizer_groups
from experiments.t10_temporal.training import streams, make_teacher, update_teacher, loss_for_batch
from run_o1_selection import build_fold, evaluate, rng_state, restore_rng
from o2_features import O2Corpus
from summer_o1_experiments import scheduler_for
from workstation_experiments import seed_all, write_json, metric_values
ROOT=Path(__file__).resolve().parents[2]
MATRIX=json.loads((ROOT/'experiments/t10_temporal/matrix.json').read_text())
SOURCES=['experiments/t10_temporal/run.py','experiments/t10_temporal/model.py','experiments/t10_temporal/training.py','experiments/t10_temporal/matrix.json','experiments/t10_temporal/checks.py','experiments/t10_temporal/audit.py','models/selective_mamba_temporal.py','models/summer_dynamic.py','models/mamba_whar.py','experiments/t08_predictive_20261114/model.py','experiments/t08_predictive_20261114/training.py','run_o1_selection.py','o2_features.py','summer_o1_experiments.py','workstation_experiments.py','run_local_message_pilot.py']
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def atomic_torch(path,obj):
    path=Path(path);tmp=path.with_name(path.name+'.tmp');torch.save(obj,tmp);os.replace(tmp,path)
def atomic_json(path,obj):
    path=Path(path);tmp=path.with_name(path.name+'.tmp');tmp.write_text(json.dumps(obj,indent=2,default=str)+'\n',encoding='utf8');os.replace(tmp,path)
@contextlib.contextmanager
def queue_lock(path):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);f=path.open('a+b')
    try:
        if os.name=='nt':
            import msvcrt
            f.seek(0,os.SEEK_END)
            if f.tell()==0:f.write(b'0');f.flush()
            f.seek(0);msvcrt.locking(f.fileno(),msvcrt.LK_NBLCK,1)
        else:
            import fcntl
            fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB)
        yield
    finally:
        try:
            if os.name=='nt':
                f.seek(0);msvcrt.locking(f.fileno(),msvcrt.LK_UNLCK,1)
            else: fcntl.flock(f,fcntl.LOCK_UN)
        finally:f.close()
def config(owner,stage,job):
    return dict(experiment='t10_temporal_r1',owner=owner,stage=stage,arm=job['arm'],temporal=job['temporal'],subject=job['subject'],seed=job['seed'],epochs=4 if stage=='smoke' else 16,batch_size=64,precision='fp32',features='mag_dot',lr=1e-4,weight_decay=5e-4,selection='source ideal fixed33 macro-F1',teacher_decay=.99,order_seed=job['seed']+10000,corruption_seed=job['seed']+20000,dropout_seed=job['seed']+30000)
def make_order_hash(order):return hashlib.sha256(order.numpy().tobytes()).hexdigest()
def run_job(fold,job,cfg,path,device):
    path.mkdir(parents=True,exist_ok=True)
    if (path/'result.json').is_file():
        old=json.loads((path/'result.json').read_text());assert old['config']==cfg,'completed result fingerprint mismatch';return old
    seed_all(cfg['seed']);model=TemporalFactorial(cfg['arm'],cfg['temporal']).to(device);teacher=make_teacher(model)
    # The independent training streams must not depend on parameter allocation.
    gens=streams(cfg['seed']);torch.manual_seed(cfg['dropout_seed']);torch.cuda.manual_seed_all(cfg['dropout_seed'])
    opt=torch.optim.Adam(optimizer_groups(model),lr=1e-4,betas=(.9,.999),eps=1e-8,fused=False,foreach=False)
    sch=scheduler_for(opt,cfg['epochs']);history=[];start_epoch=1;best=-1.;best_epoch=0
    latest=path/'latest.pt'
    if latest.exists():
        state=torch.load(latest,map_location='cpu',weights_only=False);assert state['config']==cfg,'resume config mismatch'
        model.load_state_dict(state['model']);teacher.load_state_dict(state['teacher']);opt.load_state_dict(state['optimizer']);sch.load_state_dict(state['scheduler']);restore_rng(state)
        for k,g in gens.items():g.set_state(state['streams'][k])
        history=state['history'];start_epoch=state['epoch']+1;best=state['best'];best_epoch=state['best_epoch']
    data=fold['data']['train'];n=len(data['y']);raw=fold['raw_train'];device_data={k:v for k,v in data.items()}
    for epoch in range(start_epoch,cfg['epochs']+1):
        model.train();teacher.eval();order=torch.randperm(n,generator=gens['order']);totals={};seen=correct=0;stresshash=hashlib.sha256();t0=time.perf_counter();orderhash=make_order_hash(order)
        for st in range(0,n,64):
            ix=order[st:st+64];batch={k:v[ix].to(device) for k,v in device_data.items()};batch['raw']=raw[ix].to(device)
            opt.zero_grad(set_to_none=True);loss,parts,logits=loss_for_batch(model,teacher,batch,fold,gens['corruption'],epoch,stress_hasher=stresshash)
            if not torch.isfinite(loss):raise FloatingPointError('nonfinite loss')
            loss.backward();norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True)
            if not torch.isfinite(norm):raise FloatingPointError('nonfinite grad norm')
            opt.step();update_teacher(teacher,model)
            for k,v in {'loss':loss.detach(),**parts,'grad_norm':norm}.items():totals[k]=totals.get(k,0.)+float(v)*len(ix)
            seen+=len(ix);correct+=int((logits.argmax(-1)==batch['y']).sum())
        ideal,_,_=evaluate(model,fold['data']['val_ideal'],64);selfm,_,_=evaluate(model,fold['data']['val_self'],64)
        if ideal['macro_f1']>best:
            best,best_epoch=ideal['macro_f1'],epoch;atomic_torch(path/'best.pt',dict(model=model.state_dict(),epoch=epoch,config=cfg,mean=fold['mean'],std=fold['std']))
        sch.step();row=dict(epoch=epoch,train={**{k:v/seen for k,v in totals.items()},'clean_accuracy':correct/seen,'windows':seen,'order_sha256':orderhash,'stress_sha256':stresshash.hexdigest(),'seconds':time.perf_counter()-t0},val_ideal=ideal,val_self=selfm,lr=opt.param_groups[0]['lr'],peak_cuda_allocated_mb=torch.cuda.max_memory_allocated()/2**20);history.append(row)
        state=dict(model=model.state_dict(),teacher=teacher.state_dict(),optimizer=opt.state_dict(),scheduler=sch.state_dict(),config=cfg,epoch=epoch,best=best,best_epoch=best_epoch,history=history,mean=fold['mean'],std=fold['std'],raw_scale=fold['raw_scale'],streams={k:g.get_state() for k,g in gens.items()},**rng_state());atomic_torch(latest,state)
        with (path/'epochs.jsonl').open('a',encoding='utf8') as f:f.write(json.dumps(row)+'\n')
        atomic_json(path/'heartbeat.json',dict(epoch=epoch,updated=datetime.now().astimezone().isoformat(),pid=os.getpid()))
        print(f"{cfg['temporal']} {cfg['arm']} subj{cfg['subject']} seed{cfg['seed']} epoch{epoch}/{cfg['epochs']} idealF1={ideal['macro_f1']:.4f} selfF1={selfm['macro_f1']:.4f}",flush=True)
    if not (path/'result.json').exists():
        b=torch.load(path/'best.pt',map_location='cpu',weights_only=False);model.load_state_dict(b['model']);metrics={};preds={}
        for split in ('val_ideal','val_self'):
            artifact=path/f'predictions_{split}.npz';metrics[split],_,_=evaluate(model,fold['data'][split],64,artifact,fold['metadata'][split]);preds[split]=str(artifact.name)
        result=dict(config=cfg,metrics=metrics,history=history,best_epoch=best_epoch,split=fold['summary'],prediction_files=preds,parameter_count=sum(p.numel() for p in model.parameters()))
        atomic_json(path/'result.json',result)
    return json.loads((path/'result.json').read_text())
def inventory():return {p:sha(ROOT/p) for p in SOURCES}
def data_hashes():return {p.name:sha(p) for p in sorted((ROOT/'dataset/workstation_o1').glob('*.npz'))}
def validate_t09(path):
    path=Path(path);audit=json.loads((path/'audit.json').read_text());assert audit.get('passed') and audit.get('combinations')==4 and audit.get('full_training') is False
    ref=json.loads((path/'reference.json').read_text());assert ref.get('commit')=='e9594ce1c732d97440f0332fdc43170a2294dbfa'
    for name,h in ref.get('hashes',{}).items():assert sha(path/Path(name).name)==h,'T09 reference hash mismatch '+name
    src=json.loads((path/'sources.json').read_text());src={k.replace('\\','/'):v for k,v in src.items()}
    expected={'experiments/t09_selective_ssm/check.py','models/selective_mamba_temporal.py','models/summer_dynamic.py','run_o1_selection.py'}
    assert set(src)==expected
    for name,h in src.items():assert sha(ROOT/name)==h,'T09 source mismatch '+name
    return {name:sha(path/name) for name in ('audit.json','sources.json','reference.json')}
def validate_receipt(path,owner):
    receipt=Path(path)
    assert receipt.is_file(),'smoke receipt missing'
    r=json.loads(receipt.read_text());assert r.get('passed') is True and r.get('stage')=='smoke' and r.get('owner')==owner
    assert r.get('sources')==inventory() and r.get('data')==data_hashes(),'smoke fingerprint mismatch'
    assert r.get('jobs_passed')==10,'smoke must pass all10 jobs'
def validate_reuse_source(path,owner,t09):
    old=json.loads((Path(path)/'fingerprint.json').read_text())
    assert old.get('owner')==owner and old.get('data')==data_hashes() and old.get('t09')==t09,'reuse source environment/data mismatch'
    current=inventory();previous=old['sources'];assert set(current)==set(previous)
    changed={name for name in current if current[name]!=previous[name]}
    assert changed.issubset({'experiments/t10_temporal/run.py','experiments/t10_temporal/audit.py'}),'training or check sources changed; cannot reuse smoke jobs'
    return Path(path)
class Tee:
    def __init__(self,console,log):self.console=console;self.log=log
    def write(self,value):self.console.write(value);self.log.write(value);self.log.flush()
    def flush(self):self.console.flush();self.log.flush()
def main():
    p=argparse.ArgumentParser();p.add_argument('--owner',choices=['laptop','colab','laptop_remaining'],required=True);p.add_argument('--stage',choices=['smoke','full'],required=True);p.add_argument('--output',required=True);p.add_argument('--smoke-receipt');p.add_argument('--preflight',required=True);p.add_argument('--t09-output',required=True);p.add_argument('--reference-dir');p.add_argument('--reuse-smoke-jobs');a=p.parse_args()
    out=Path(a.output).resolve();fresh=not out.exists();out.mkdir(parents=True,exist_ok=True);device='cuda'
    logfile=(out/'console.log').open('a',encoding='utf8');sys.stdout=Tee(sys.stdout,logfile);sys.stderr=Tee(sys.stderr,logfile)
    with queue_lock(out/'.queue.lock'):
        started=time.perf_counter();atomic_json(out/'status.json',dict(state='running',stage=a.stage,owner=a.owner))
        try:
            assert torch.cuda.is_available();torch.set_num_threads(4);torch.set_num_interop_threads(2);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;torch.backends.cudnn.benchmark=False
            pre=json.loads(Path(a.preflight).read_text());assert pre.get('passed') and pre.get('owner')==a.owner and pre.get('jobs')==10 and pre.get('sources')==inventory() and pre.get('data')==data_hashes(),'preflight gate/fingerprint missing or failed'
            t09=validate_t09(a.t09_output)
            reuse=validate_reuse_source(a.reuse_smoke_jobs,a.owner,t09) if a.reuse_smoke_jobs else None
            if a.stage=='full':validate_receipt(a.smoke_receipt,a.owner)
            src=inventory();data=data_hashes();fingerprint=dict(sources=src,data=data,owner=a.owner,t09=t09)
            if (out/'fingerprint.json').exists():assert json.loads((out/'fingerprint.json').read_text())==fingerprint,'output fingerprint mismatch; preserve and use fresh attempt'
            else:
                assert fresh,'output exists without fingerprint; preserve it and choose a fresh attempt folder'
                atomic_json(out/'fingerprint.json',fingerprint)
            for name,h in src.items():
                target=out/'source'/name;target.parent.mkdir(parents=True,exist_ok=True)
                if target.exists():assert sha(target)==h,'saved source snapshot mismatch'
                else:target.write_bytes((ROOT/name).read_bytes())
                assert sha(target)==h
            atomic_json(out/'environment.json',dict(python=sys.version,torch=torch.__version__,cuda=torch.version.cuda,gpu=torch.cuda.get_device_name(),device=device,precision='fp32',stage=a.stage,owner=a.owner))
            corpus=O2Corpus();selected=[j for j in MATRIX['jobs'] if j['owner']==('colab' if a.owner=='laptop_remaining' else a.owner)]
            if a.stage=='smoke':selected=[dict(subject=2,seed=13,temporal=t,arm=arm,owner=a.owner,id=f'{t}__{arm}__subject_2__seed_13') for t in TEMPORALS for arm in ARMS]
            assert len(selected)==(10 if a.stage=='smoke' else (10 if a.owner=='laptop' else 30))
            done=[];folds={}
            for j in selected:
                sid=j['subject'];
                if sid not in folds:
                    f=build_fold(corpus,sid);ix=f['indices']['train'];f['raw_train']=corpus.x[ix];f['data']['train']['raw']=f['raw_train'];f['raw_scale']=f['raw_train'].std((0,1)).reshape(9,2,3).square().mean(-1).sqrt().clamp_min(1e-4).repeat_interleave(3,-1)
                    f['metadata']={k:dict(corpus_indices=ids,subjects=corpus.subjects[ids],recordings=corpus.sources[ids],scenarios=corpus.scenarios[ids],window_starts=corpus.starts[ids].numpy()) for k,ids in f['indices'].items()};folds[sid]=f;atomic_json(out/f'fold_{sid:02d}.json',f['summary']);atomic_torch(out/f'normalization_{sid:02d}.pt',dict(mean=f['mean'],std=f['std'],indices=f['indices']))
                cfg=config(a.owner,a.stage,j);cfg['validation_subject']=folds[sid]['summary']['validation_subject'];jobpath=out/'jobs'/j['id']
                if reuse and not (jobpath/'result.json').exists():
                    prior=reuse/'jobs'/j['id'];prior_result=json.loads((prior/'result.json').read_text());assert prior_result['config']==cfg,'reused smoke job config mismatch'
                    assert all((prior/n).is_file() for n in ('latest.pt','best.pt','epochs.jsonl','predictions_val_ideal.npz','predictions_val_self.npz'))
                    jobpath.parent.mkdir(parents=True,exist_ok=True);shutil.copytree(prior,jobpath)
                res=run_job(folds[sid],j,cfg,jobpath,device);done.append(j['id']);atomic_json(out/'progress.json',dict(completed=done,total=len(selected)))
            assert inventory()==src and data_hashes()==data
            from experiments.t10_temporal.audit import audit
            audit_result=audit(out,a.owner,a.stage);atomic_json(out/'audit.json',audit_result);assert audit_result['passed']
            receipt=dict(passed=True,stage=a.stage,owner=a.owner,jobs_passed=len(done),job_ids=done,sources=src,data=data,t09=t09,seconds=time.perf_counter()-started,gpu=torch.cuda.get_device_name(),audit=audit_result)
            atomic_json(out/'receipt.json',receipt);atomic_json(out/'status.json',dict(state='complete',**receipt))
        except BaseException as e:
            with (out/'error.txt').open('a',encoding='utf8') as f:f.write('\n--- attempt failure ---\n'+traceback.format_exc());atomic_json(out/'status.json',dict(state='failed',error=repr(e),elapsed=time.perf_counter()-started));raise
if __name__=='__main__':main()
