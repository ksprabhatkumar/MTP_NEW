"""T11 fixed workstation queue. Smoke never starts full automatically."""
import argparse,json,sys,time,traceback,os
from pathlib import Path
from datetime import datetime
import torch
from experiments.t12_representation.model import TemporalFactorial,optimizer_groups
from experiments.t12_representation.training import streams,make_teacher,loss_for_batch,update_teacher
from experiments.t12_representation.common import *
from experiments.t12_representation.data import build_fold,evaluate
from run_o1_selection import rng_state,restore_rng
from o2_features import O2Corpus
from summer_o1_experiments import scheduler_for
from workstation_experiments import seed_all
import hashlib

def make_order_hash(order):return hashlib.sha256(order.numpy().tobytes()).hexdigest()
def config(stage,job):
    return dict(experiment='t12_representation_r1',owner='workstation',stage=stage,arm=job['arm'],temporal=job['temporal'],subject=job['subject'],validation_subject=job['validation_subject'],seed=job['seed'],epochs=4 if stage=='smoke' else 31,batch_size=64,precision='fp32',features=representation(),lr=1e-4,weight_decay=5e-4,selection='source ideal fixed33 macro-F1',teacher_decay=.99,order_seed=job['seed']+10000,corruption_seed=job['seed']+20000,dropout_seed=job['seed']+30000)

def run_job(fold,job,cfg,path,device):
    path.mkdir(parents=True,exist_ok=True)
    if (path/'result.json').is_file():
        old=json.loads((path/'result.json').read_text());assert old['config']==cfg,'completed result fingerprint mismatch';return old
    seed_all(cfg['seed']);model=TemporalFactorial(cfg['arm'],cfg['temporal']).to(device);teacher=make_teacher(model)
    torch.cuda.reset_peak_memory_stats()
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
        # Deterministic, explicitly bounded probe for later graph/gate plots.
        with torch.inference_mode():
            probe=fold['data']['val_self'];_,_,diag=model(probe['x'][:64].cuda(),probe['a'][:64].cuda(),True)
            atomic_torch(path/'mechanism_probe.pt',dict(corpus_indices=fold['indices']['val_self'][:64],availability=probe['a'][:64],diagnostics={k:v.cpu() for k,v in diag.items()},scope='first up to 64 source-self validation windows; descriptive only'))
        result=dict(config=cfg,metrics=metrics,history=history,best_epoch=best_epoch,split=fold['summary'],prediction_files=preds,parameter_count=sum(p.numel() for p in model.parameters()))
        atomic_json(path/'result.json',result)
    return json.loads((path/'result.json').read_text())

def main():
    p=argparse.ArgumentParser();p.add_argument('--stage',choices=['smoke','full'],required=True)
    p.add_argument('--output',required=True);p.add_argument('--preflight',required=True)
    p.add_argument('--smoke-receipt');p.add_argument('--confirm-full')
    a=p.parse_args();configure();out=Path(a.output).resolve()
    if a.stage=='full':
        assert a.confirm_full=='SMOKE_REVIEWED_BY_USER','Return smoke to user before full training'
    fp=fingerprint();pre=json.loads(Path(a.preflight).read_text())
    assert pre['passed'] and pre['cases']==20 and pre['fingerprint']==fp,'Preflight stale/failed'
    smoke_hash=None
    if a.stage=='full':
        receipt=Path(a.smoke_receipt);s=json.loads(receipt.read_text())
        assert s['passed'] and s['stage']=='smoke' and s['jobs_passed']==20 and s['fingerprint']==fp
        assert set(s['job_ids'])=={j['id'] for j in selected('smoke')}
        assert json.loads((receipt.parent/'audit.json').read_text())['passed']
        smoke_hash=sha(receipt)
    # Global project device lock prevents smoke/full processes in different
    # output directories from competing. Do not delete/bypass it.
    with queue_lock(ROOT/'results/.t11_device.lock'):
        fresh=not out.exists();out.mkdir(parents=True,exist_ok=True)
        if (out/'fingerprint.json').exists():assert json.loads((out/'fingerprint.json').read_text())==fp,'Resume fingerprint changed'
        else:
            assert fresh,'Existing output without fingerprint; use a fresh attempt'
            atomic_json(out/'fingerprint.json',fp)
        with queue_lock(out/'.queue.lock'),(out/'console.log').open('a',encoding='utf8') as log,console_log(log):
            started=time.perf_counter()
            try:
                atomic_json(out/'status.json',dict(state='running',stage=a.stage,pid=os.getpid()))
                atomic_json(out/'environment.json',environment())
                atomic_json(out/'command.json',dict(argv=sys.argv,smoke_receipt_sha256=smoke_hash))
                for name,h in fp['sources'].items():
                    target=out/'source'/name;target.parent.mkdir(parents=True,exist_ok=True)
                    if target.exists():assert sha(target)==h
                    else:target.write_bytes((ROOT/name).read_bytes())
                corpus=O2Corpus();done=[];fold=None;sid=None
                for job in selected(a.stage):
                    if sid!=job['subject']:
                        if fold is not None:del fold
                        sid=job['subject'];fold=build_fold(corpus,sid)
                        atomic_json(out/f'fold_{sid:02d}.json',fold['summary'])
                        atomic_torch(out/f'normalization_{sid:02d}.pt',dict(mean=fold['mean'],std=fold['std'],indices=fold['indices']))
                    run_job(fold,job,config(a.stage,job),out/'jobs'/job['id'],'cuda')
                    done.append(job['id']);atomic_json(out/'progress.json',dict(completed=done,total=len(selected(a.stage))))
                    torch.cuda.empty_cache()
                del fold,corpus
                assert fingerprint()==fp,'Sources/data/environment changed during queue'
                from experiments.t12_representation.audit import audit
                result=audit(out,'workstation',a.stage);atomic_json(out/'audit.json',result);assert result['passed']
                from experiments.t12_representation.report import report
                report(out,a.stage)
                if a.stage=='full':
                    from experiments.t12_representation.targets import evaluate_targets
                    evaluate_targets(out)
                receipt=dict(passed=True,stage=a.stage,jobs_passed=len(done),job_ids=done,fingerprint=fp,audit=result,seconds=time.perf_counter()-started)
                atomic_json(out/'receipt.json',receipt);atomic_json(out/'status.json',dict(state='complete',**receipt))
            except BaseException:
                error=traceback.format_exc()
                with (out/'error.txt').open('a',encoding='utf8') as f:f.write(error+'\n')
                atomic_json(out/'status.json',dict(state='failed',stage=a.stage,error=error,seconds=time.perf_counter()-started));raise

if __name__=='__main__':main()
