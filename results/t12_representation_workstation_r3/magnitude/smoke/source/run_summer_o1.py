"""Verified summer-source O1 backend: preflight, smoke, source pilot, full suite."""
import argparse
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import threading
import time
import traceback

ROOT = Path(__file__).resolve().parent
os.chdir(ROOT)
SOURCE_FILES = ['run_summer_o1.py','summer_o1_experiments.py','models/summer_dynamic.py',
    'models/mamba_whar.py','o2_features.py','o2_experiments.py','workstation_experiments.py',
    'prepare_workstation_data.py','baselines/dynamicwhar_adapter.py','run_training.py']


def status_write(output, data):
    from workstation_experiments import write_json
    data = dict(data, pid=os.getpid(), updated=datetime.now().astimezone().isoformat())
    write_json(output/'status.json',data)
    text = ('SUMMER-BASED O1 HANDOFF\n'+json.dumps(data,indent=2)+'\n\n'
        f'Workspace: {ROOT}\nOutput: {output}\n'
        'Authoritative checklist: entire_plan.md. Model provenance: baselines/summer_original/PROVENANCE.json.\n'
        'Main CE; GAE is a separate masked-target reconstruction ablation.\n'
        'No graph is a required control; the target is residual dynamic graph.\n'
        'Status is a snapshot. Inspect status.json, epochs.jsonl and GPU telemetry.\n'
        'Resume through run_long_training.py with the SAME parent output directory.\n'
        'It skips completed stages/jobs; the backend restores model, optimizer, scheduler and RNG.\n'
        'Do not launch another writer or modify source during an active run.\n'
        'After completion: compare static_sum, normalized static, residual and no graph, then\n'
        'augmentation/GAE/branch-balanced controls; examine recording coverage and interventions.\n'
        'All subjects already inspected in prior runs: results are exploratory.\n')
    (output/'TRAINING_HANDOFF.txt').write_text(text,encoding='utf-8')
    (ROOT/'SUMMER_O1_HANDOFF.txt').write_text(text,encoding='utf-8')
    (ROOT/'current_step_update.txt').write_text(text,encoding='utf-8')


def run(args):
    import msvcrt
    import numpy as np
    import torch
    from o2_features import O2Corpus
    from summer_o1_experiments import EXPERIMENTS, model_for, optimizer_for, train_epoch, run_job, summarize
    from o2_experiments import evaluate
    from workstation_experiments import write_json, seed_all
    from run_training import telemetry, Tee
    output = Path(args.output).resolve()
    output.mkdir(parents=True,exist_ok=True)
    lock = (output/'.run.lock').open('a+b')
    if lock.tell()==0: lock.write(b'0'); lock.flush()
    lock.seek(0)
    msvcrt.locking(lock.fileno(),msvcrt.LK_NBLCK,1)
    log = (output/'run.log').open('a',encoding='utf-8',buffering=1)
    sys.stdout,sys.stderr = Tee(sys.stdout,log),Tee(sys.stderr,log)
    stop = threading.Event()
    monitor = None
    try:
        if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
            raise RuntimeError('CUDA BF16 GPU required')
        torch.set_num_threads(8); torch.set_num_interop_threads(2)
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        print('SUMMER O1',args.mode,output,torch.cuda.get_device_name(),flush=True)
        paths = [ROOT/p for p in SOURCE_FILES] + list((ROOT/'tests').glob('*.py'))
        paths += list((ROOT/'baselines/summer_original').rglob('*.py'))
        paths += list((ROOT/'baselines/DynamicWHAR').rglob('*.py'))
        hashes = {str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
        fingerprint = {'sources':hashes,'cache_audit':hashlib.sha256((ROOT/'dataset/workstation_o1/audit.json').read_bytes()).hexdigest(),
            'provenance':json.loads((ROOT/'baselines/summer_original/PROVENANCE.json').read_text(encoding='utf-8-sig'))}
        if (output/'fingerprint.json').exists() and json.loads((output/'fingerprint.json').read_text())!=fingerprint:
            raise ValueError('Run source/data provenance changed')
        write_json(output/'fingerprint.json',fingerprint)
        if not (output/'source').exists():
            for p in paths:
                target=output/'source'/p.relative_to(ROOT); target.parent.mkdir(parents=True,exist_ok=True)
                shutil.copy2(p,target)
            shutil.copy2(ROOT/'entire_plan.md',output/'entire_plan_at_start.md')
        write_json(output/'system.json',{'python':sys.version,'executable':sys.executable,'torch':torch.__version__,
            'cuda':torch.version.cuda,'gpu':torch.cuda.get_device_name(),'batch_size':args.batch_size,
            'nondeterminism':'fixed seeds; optimized kernels may remain nondeterministic'})
        with (output/'requirements.lock.txt').open('w') as req:
            subprocess.run([sys.executable,'-m','pip','freeze'],stdout=req,check=True)
        monitor=threading.Thread(target=telemetry,args=(output,stop),daemon=True); monitor.start()
        status_write(output,{'state':args.mode,'plan_step':'P4.3' if args.mode=='preflight' else 'P4.4'})
        corpus=O2Corpus()
        folds={}
        for subject in (2,5,15):
            fold=corpus.fold(subject,'raw')
            summary=fold['summary']
            assert subject not in summary['train_subjects']
            assert summary['validation_subject'] not in summary['train_subjects']
            assert not set(summary['recordings']['train']) & set(summary['recordings']['ideal'])
            folds[str(subject)]=summary
            del fold; torch.cuda.empty_cache()
        write_json(output/'fold_preflight.json',folds)
        if args.mode=='preflight':
            test = subprocess.run([sys.executable,'-m','unittest','discover','-s','tests','-v'],capture_output=True,text=True)
            (output/'unit_tests.log').write_text(test.stdout+test.stderr,encoding='utf-8')
            if test.returncode: raise RuntimeError('Unit tests failed')
            write_json(output/'passed.json',{'passed':True,'subjects':[2,5,15],'provenance':fingerprint['provenance']})
        elif args.mode in ('smoke','pilot'):
            fold=corpus.fold(2,'raw')
            fold['sources']={key:corpus.sources[ix] for key,ix in fold['indices'].items()}
            conditions=EXPERIMENTS if args.mode=='smoke' else [c for c in EXPERIMENTS if c.name in ('static_sum','physical','residual')]
            for condition in conditions:
                run_job(fold,condition,2,13,2 if args.mode=='smoke' else 6,args.batch_size,
                    output/condition.name,limit=2 if args.mode=='smoke' else None,
                    validation_only=args.mode=='pilot')
            write_json(output/'passed.json',{'passed':True,'mode':args.mode,'conditions':[c.name for c in conditions],
                'research_result':False if args.mode=='smoke' else 'source-validation pilot only'})
        else:
            jobs=[{'condition':c.name,'subject':s,'seed':seed} for s in (2,5,15) for c in EXPERIMENTS for seed in (13,29,47)]
            schedule_file=output/'schedule.json'
            if schedule_file.exists():
                schedule=json.loads(schedule_file.read_text())
                if schedule['jobs']!=jobs or schedule['batch_size']!=args.batch_size: raise ValueError('Schedule changed')
            else:
                calibration={}
                fold=corpus.fold(2,'raw')
                for condition in EXPERIMENTS:
                    seed_all(101); model=model_for(condition); optimizer=optimizer_for(model)
                    train_epoch(model,optimizer,fold,condition,args.batch_size,limit=2)
                    start=time.perf_counter()
                    train_epoch(model,optimizer,fold,condition,args.batch_size)
                    evaluate(model,fold,'val_ideal',args.batch_size)
                    calibration[condition.name]=time.perf_counter()-start
                    print('CALIBRATION',condition.name,calibration[condition.name],flush=True)
                    del model,optimizer; torch.cuda.empty_cache()
                del fold
                epoch_sum=sum(calibration[j['condition']] for j in jobs)
                reserve=epoch_sum*2.5
                epochs=max(20,min(50,int((args.hours*3600-reserve)/(epoch_sum*1.12))))
                schedule={'jobs':jobs,'epochs':epochs,'batch_size':args.batch_size,
                    'calibration':calibration,'target_hours':args.hours,
                    'estimated_hours':(epoch_sum*epochs*1.12+reserve)/3600,
                    'selection':'source ideal validation only','main_loss':'cross entropy',
                    'note':'calibrated 20-50 epochs; estimate may exceed target if minimum required budget cannot fit'}
                write_json(schedule_file,schedule)
            print('LOCKED SCHEDULE',len(jobs),'jobs',schedule['epochs'],'epochs',schedule['estimated_hours'],'hours',flush=True)
            lookup={c.name:c for c in EXPERIMENTS}
            for number,job in enumerate(jobs,1):
                destination=output/'jobs'/job['condition']/f'subject_{job["subject"]:02d}'/f'seed_{job["seed"]}'
                if (destination/'result.json').exists(): continue
                fold=corpus.fold(job['subject'],'raw')
                fold['sources']={key:corpus.sources[ix] for key,ix in fold['indices'].items()}
                split_dir=output/'folds'/f'subject_{job["subject"]:02d}'; split_dir.mkdir(parents=True,exist_ok=True)
                if not (split_dir/'indices.npz').exists():
                    np.savez_compressed(split_dir/'indices.npz',**fold['indices'],mean=fold['mean'].numpy(),std=fold['std'].numpy())
                    write_json(split_dir/'manifest.json',fold['summary'])
                status_write(output,{'state':'training','plan_step':'P4.5','current_job':number,'total_jobs':len(jobs),
                    'job':job,'epochs':schedule['epochs'],'estimated_hours':schedule['estimated_hours']})
                run_job(fold,lookup[job['condition']],job['subject'],job['seed'],schedule['epochs'],args.batch_size,destination)
                summarize(output)
                del fold; torch.cuda.empty_cache()
            write_json(output/'completed.json',{'completed_jobs':len(jobs),'finished':datetime.now().isoformat()})
        status_write(output,{'state':'complete','mode':args.mode,'plan_step':'P4.7' if args.mode=='full' else 'P4.4'})
    except BaseException as error:
        status_write(output,{'state':'failed','mode':args.mode,'error':repr(error),'traceback':traceback.format_exc()})
        raise
    finally:
        stop.set()
        if monitor: monitor.join(timeout=12)
        lock.close()


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--mode',choices=('preflight','smoke','pilot','full'),required=True)
    p.add_argument('--output',required=True)
    p.add_argument('--hours',type=float,default=5.5)
    p.add_argument('--batch-size',type=int,default=64)
    args=p.parse_args()
    if args.hours<=0 or args.batch_size<2: p.error('Invalid budget or batch size')
    run(args)
