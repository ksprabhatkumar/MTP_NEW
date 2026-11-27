"""Run both representation queues sequentially; never auto-promote smoke to full."""
import argparse,json,os,subprocess,sys,time,traceback
from pathlib import Path
from .common import ROOT,queue_lock,atomic_json,configure,fingerprint,selected

def main():
    p=argparse.ArgumentParser();p.add_argument('--stage',choices=['smoke','full'],required=True)
    p.add_argument('--output-root',required=True);p.add_argument('--detach',action='store_true');p.add_argument('--confirm-full')
    a=p.parse_args();root=Path(a.output_root).resolve();root.mkdir(parents=True,exist_ok=True)
    if a.stage=='full':assert a.confirm_full=='SMOKE_REVIEWED_BY_USER','Separate smoke review required'
    if a.detach:
        args=[sys.executable,'-u','-m','experiments.t12_representation.queue','--stage',a.stage,'--output-root',str(root)]
        if a.confirm_full:args+=['--confirm-full',a.confirm_full]
        options=dict(cwd=ROOT,stdin=subprocess.DEVNULL)
        if os.name=='nt':options['creationflags']=subprocess.DETACHED_PROCESS|subprocess.CREATE_NEW_PROCESS_GROUP
        else:options['start_new_session']=True
        log=root/f'{a.stage}_batch_{time.time_ns()}.log'
        with log.open('x',encoding='utf8') as f:child=subprocess.Popen(args,stdout=f,stderr=subprocess.STDOUT,**options)
        atomic_json(log.with_suffix('.json'),dict(pid=child.pid,command=args,log=str(log)))
        print(json.dumps(dict(pid=child.pid,log=str(log))));return
    with queue_lock(ROOT/'results/.t12_batch.lock'):
        status=root/f'{a.stage}_batch_status.json'
        try:
            # Check BOTH smoke receipts before starting any full training.
            if a.stage=='full':
                configure()
                for rep in ('raw','magnitude'):
                    os.environ['MTP_T12_REPRESENTATION']=rep;base=root/rep
                    assert not (base/'full').exists(),'Existing full attempt: stop and request reviewed recovery; do not duplicate'
                    receipt=json.loads((base/'smoke/receipt.json').read_text())
                    pre=json.loads((base/'preflight/preflight.json').read_text())
                    assert receipt['passed'] and receipt['jobs_passed']==20 and pre['passed'] and pre['cases']==20
                    assert set(receipt['job_ids'])=={j['id'] for j in selected('smoke')}
                    assert receipt['fingerprint']==pre['fingerprint']==fingerprint()
                    assert json.loads((base/'smoke/audit.json').read_text())['passed']
            completed=[]
            for rep in ('raw','magnitude'):
                args=[sys.executable,'-u','-m','experiments.t12_representation.launch','--representation',rep,'--stage',a.stage,'--output-base',str(root/rep)]
                if a.confirm_full:args+=['--confirm-full',a.confirm_full]
                child=subprocess.Popen(args,cwd=ROOT)
                atomic_json(status,dict(state='running',representation=rep,pid=os.getpid(),child_pid=child.pid,completed=completed,command=args))
                rc=child.wait();assert rc==0,f'{rep} supervisor exit {rc}; stop'
                receipt=json.loads((root/rep/a.stage/'receipt.json').read_text());assert receipt['passed']
                completed.append(rep)
            atomic_json(status,dict(state='complete',stage=a.stage,completed=completed,full_authorized=a.stage=='full'))
        except BaseException:
            atomic_json(status,dict(state='failed',stage=a.stage,error=traceback.format_exc()));raise

if __name__=='__main__':main()
