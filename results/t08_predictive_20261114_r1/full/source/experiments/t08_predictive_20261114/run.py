"""CPU-staged source-validation repair pilot; no held-out test evaluation."""
import argparse,hashlib,json,os,shutil,sys,time,traceback
from datetime import datetime
from pathlib import Path
import numpy as np
import torch
from experiments.t08_predictive_20261114.model import ARMS,PredictiveRepair as SummerGraphRepair
from experiments.t08_predictive_20261114.training import streams,make_teacher,update_teacher,loss_for_batch
from run_o1_selection import build_fold,evaluate,rng_state,restore_rng
from o2_features import O2Corpus
from summer_o1_experiments import scheduler_for
from workstation_experiments import seed_all,save_checkpoint,write_json

ROOT=Path(__file__).resolve().parents[2]
SOURCES=["experiments/t08_predictive_20261114/model.py","experiments/t08_predictive_20261114/training.py","experiments/t08_predictive_20261114/run.py","experiments/t08_predictive_20261114/checks.py","experiments/t08_predictive_20261114/analyze.py","experiments/t08_predictive_20261114/review.py","experiments/t08_predictive_20261114/launch.ps1","experiments/t08_predictive_20261114/spec.md","models/summer_dynamic.py","models/mamba_whar.py","run_o1_selection.py","models/summer_gated_neighbor.py","models/summer_message_correction.py","o2_features.py","o2_experiments.py","summer_o1_experiments.py","workstation_experiments.py","prepare_workstation_data.py","run_local_message_pilot.py","requirements_local_gpu.txt"]

def training_epoch(model,teacher,opt,fold,batch_size,generators,epoch,job):
    model.train();teacher.eval();data=fold['data']['train']
    save_checkpoint(job/f'rng_epoch_{epoch:02d}.pt',{k:v.get_state() for k,v in generators.items()})
    order=torch.randperm(len(data['y']),generator=generators['order'])
    totals={};count=correct=0;started=time.perf_counter()
    for start in range(0,len(order),batch_size):
        ix=order[start:start+batch_size];batch={k:v[ix].cuda() for k,v in data.items()}
        opt.zero_grad(set_to_none=True)
        loss,components,logits=loss_for_batch(model,teacher,batch,fold,generators['corruption'],epoch)
        if not torch.isfinite(loss):raise FloatingPointError('Nonfinite training loss')
        loss.backward()
        norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True)
        opt.step();update_teacher(teacher,model)
        items=dict(loss=loss.detach(),grad_norm=norm,**components)
        for key,value in items.items():totals[key]=totals.get(key,0.)+float(value)*len(ix)
        count+=len(ix);correct+=int((logits.argmax(-1)==batch['y']).sum())
    torch.cuda.synchronize()
    return {**{k:v/count for k,v in totals.items()},'clean_accuracy':correct/count,'seconds':time.perf_counter()-started,
            'windows':count,'order_sha256':hashlib.sha256(order.numpy().tobytes()).hexdigest()}

def run_job(fold,arm,subject,args,job):
    job.mkdir(parents=True,exist_ok=True)
    config=dict(experiment='t08_predictive_20261114_r1',arm=arm,subject=subject,validation_subject=fold['summary']['validation_subject'],seed=args.seed,
        epochs=args.epochs,batch_size=args.batch_size,hidden=16,features='mag_dot',lr=1e-4,weight_decay=5e-4,
        selection='source ideal fixed33 macro-F1',teacher_decay=.99,warmup_epochs=2,
        loss_weights=dict(consistency=.5,repair=.1,quality=.1),temperature=2.,ramp_epochs=3,
        corruption=dict(type='unit-free predictive repair gain/drift proxy',window_probability=.75,sensors=[1,7],
                        modality_gain=[.6,1.4],drift_scale=.30,knots=4),
        precision='bf16' if torch.cuda.is_bf16_supported() else 'fp32',evaluation_model='student')
    if (job/'result.json').exists():
        result=json.loads((job/'result.json').read_text());assert result['config']==config
        return result
    seed_all(args.seed);model=SummerGraphRepair(arm).cuda();teacher=make_teacher(model)
    generators=streams(args.seed)
    # Data/corruption streams are independent; reset dropout after module allocation.
    torch.manual_seed(args.seed+30000);torch.cuda.manual_seed_all(args.seed+30000)
    opt=torch.optim.Adam(model.parameters(),lr=1e-4,weight_decay=5e-4,fused=True)
    scheduler=scheduler_for(opt,args.epochs)
    history=[];first=1;best=-1.;best_epoch=0
    if (job/'latest.pt').exists():
        state=torch.load(job/'latest.pt',map_location='cpu',weights_only=False);assert state['config']==config
        model.load_state_dict(state['model']);teacher.load_state_dict(state['teacher']);opt.load_state_dict(state['optimizer']);scheduler.load_state_dict(state['scheduler'])
        restore_rng(state)
        for k,v in generators.items():v.set_state(state['streams'][k])
        history=state['history'];first=state['epoch']+1;best=state['best'];best_epoch=state['best_epoch']
        del state
    write_json(job/'config.json',config)
    (job/'epochs.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in history))
    started=time.perf_counter();torch.cuda.reset_peak_memory_stats()
    for epoch in range(first,args.epochs+1):
        train=training_epoch(model,teacher,opt,fold,args.batch_size,generators,epoch,job)
        vi,_,_=evaluate(model,fold['data']['val_ideal'],args.batch_size)
        vs,_,_=evaluate(model,fold['data']['val_self'],args.batch_size)
        if vi['macro_f1']>best:
            best=vi['macro_f1'];best_epoch=epoch
            save_checkpoint(job/'best.pt',dict(model=model.state_dict(),config=config,epoch=epoch,mean=fold['mean'],std=fold['std']))
        scheduler.step()
        row=dict(epoch=epoch,train=train,val_ideal=vi,val_self=vs,lr=opt.param_groups[0]['lr'],
                 peak_cuda_allocated_mb=torch.cuda.max_memory_allocated()/1024**2)
        history.append(row)
        save_checkpoint(job/'latest.pt',dict(model=model.state_dict(),teacher=teacher.state_dict(),optimizer=opt.state_dict(),
            scheduler=scheduler.state_dict(),config=config,epoch=epoch,best=best,best_epoch=best_epoch,history=history,
            mean=fold['mean'],std=fold['std'],raw_scale=fold['raw_scale'],streams={k:v.get_state() for k,v in generators.items()},**rng_state()))
        with (job/'epochs.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')
        write_json(job/'heartbeat.json',dict(pid=os.getpid(),epoch=epoch,time=datetime.now().astimezone().isoformat()))
        print(f'{arm} fold{subject} epoch{epoch}/{args.epochs} loss={train["loss"]:.4f} ideal_acc={vi["accuracy"]:.4f} self_acc={vs["accuracy"]:.4f} train={train["seconds"]:.1f}s [checkpoint saved]',flush=True)
    state=torch.load(job/'best.pt',map_location='cpu',weights_only=False);model.load_state_dict(state['model'])
    metrics={}
    for split in ('val_ideal','val_self'):
        metrics[split],_,_=evaluate(model,fold['data'][split],args.batch_size,job/f'predictions_{split}.npz',fold['metadata'][split])
    with torch.inference_mode():
        sample=fold['data']['val_self'];x=sample['x'][:16].cuda();a=sample['a'][:16].cuda()
        from run_local_message_pilot import precision_context
        with precision_context():_,_,d=model(x,a,return_diagnostics=True)
        np.savez_compressed(job/'graph_samples.npz',**{k:d[k].float().cpu().numpy() for k in ('adjacency','null_mass','quality','gate','prediction')},corpus_indices=fold['indices']['val_self'][:16])
        diagnostics={k+'_rms':float(d[k].float().square().mean().sqrt()) for k in ('local','message','correction','prediction')}
    interventions={}
    if arm=='graph_repair':
        for mode in ('zero_message','zero_correction','window_shuffle'):
            model.message_intervention=mode
            interventions[mode]=evaluate(model,fold['data']['val_self'],args.batch_size)[0]
        model.message_intervention=None
    result=dict(config=config,metrics=metrics,history=history,best_epoch=best_epoch,diagnostics=diagnostics,
                interventions=interventions,split=fold['summary'],parameter_count=sum(p.numel() for p in model.parameters()),
                seconds_this_invocation=time.perf_counter()-started)
    write_json(job/'result.json',result)
    del model,teacher,opt,scheduler;torch.cuda.empty_cache()
    return result

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',required=True);parser.add_argument('--epochs',type=int,default=16)
    parser.add_argument('--subjects',default='2,5');parser.add_argument('--seeds',default='13,29');parser.add_argument('--batch-size',type=int,default=64)
    args=parser.parse_args();subjects=[int(x) for x in args.subjects.split(',')];seeds=[int(x) for x in args.seeds.split(',')]
    assert args.epochs>=4 and args.batch_size>=2 and set(subjects)<={2,5}
    assert len(set(subjects))==len(subjects) and len(set(seeds))==len(seeds) and torch.cuda.is_available()
    torch.set_num_threads(4);torch.set_num_interop_threads(2);torch.backends.cudnn.benchmark=False;torch.backends.cuda.matmul.allow_tf32=True
    output=Path(args.output).resolve();output.mkdir(parents=True,exist_ok=True)
    import msvcrt
    with (output/'.run.lock').open('a+b') as lock:
        if lock.tell()==0:lock.write(b'0');lock.flush()
        lock.seek(0);msvcrt.locking(lock.fileno(),msvcrt.LK_NBLCK,1)
        expected=len(subjects)*len(seeds)*len(ARMS)
        def status(state,**extra):write_json(output/'status.json',dict(state=state,pid=os.getpid(),updated=datetime.now().astimezone().isoformat(),expected_jobs=expected,**extra))
        try:
            hashes={name:hashlib.sha256((ROOT/name).read_bytes()).hexdigest() for name in SOURCES}
            preflight=json.loads((output/'preflight.json').read_text());assert preflight['passed'] and preflight['source_hashes']==hashes
            fingerprint=dict(sources=hashes,data={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted((ROOT/'dataset/workstation_o1').glob('*.npz'))},
                             schedule=dict(subjects=subjects,seeds=seeds,arms=list(ARMS),epochs=args.epochs,batch_size=args.batch_size))
            if (output/'fingerprint.json').exists():assert json.loads((output/'fingerprint.json').read_text())==fingerprint,'Resume fingerprint mismatch'
            write_json(output/'fingerprint.json',fingerprint)
            for name in SOURCES:
                target=output/'source'/name;target.parent.mkdir(parents=True,exist_ok=True)
                if not target.exists():shutil.copy2(ROOT/name,target)
            write_json(output/'system.json',dict(python=sys.version,executable=sys.executable,torch=torch.__version__,cuda=torch.version.cuda,gpu=torch.cuda.get_device_name()))
            status('loading');corpus=O2Corpus();count=0
            for subject in subjects:
                fold=build_fold(corpus,subject)
                raw=corpus.x[fold['indices']['train']];fold['data']['train']['raw']=raw
                fold['raw_scale']=raw.std((0,1)).reshape(9,2,3).square().mean(-1).sqrt().clamp_min(1e-4).repeat_interleave(3,-1)
                fold['metadata']={key:dict(corpus_indices=ix,subjects=corpus.subjects[ix],recordings=corpus.sources[ix],scenarios=corpus.scenarios[ix],window_starts=corpus.starts[ix].numpy()) for key,ix in fold['indices'].items()}
                write_json(output/f'fold_{subject:02d}.json',fold['summary'])
                save_checkpoint(output/f'normalization_{subject:02d}.pt',dict(mean=fold['mean'],std=fold['std'],raw_scale=fold['raw_scale'],indices=fold['indices']))
                for seed in seeds:
                    args.seed=seed
                    for arm in ARMS:
                        status('training',completed_jobs=count,subject=subject,seed=seed,arm=arm)
                        print(f'JOB {count+1}/{expected}: {arm}, excluded subject{subject}, seed{seed}',flush=True)
                        run_job(fold,arm,subject,args,output/'jobs'/arm/f'subject_{subject:02d}_seed_{seed}');count+=1
                del fold,raw
            assert all(hashlib.sha256((ROOT/n).read_bytes()).hexdigest()==h for n,h in hashes.items())
            from experiments.t08_predictive_20261114.analyze import analyze,decision
            analyze(output,expected)
            decision(output)
            write_json(output/'completed.json',dict(completed_jobs=count,expected_jobs=expected));status('complete',completed_jobs=count)
        except BaseException as exc:
            (output/'error.txt').write_text(traceback.format_exc());status('failed',error=repr(exc));raise

if __name__=='__main__':main()
