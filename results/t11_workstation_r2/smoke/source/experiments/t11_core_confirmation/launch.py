"""Durable workstation supervisor. Smoke and full are separate invocations."""
import argparse,json,os,subprocess,sys,time,traceback
from pathlib import Path
from .common import ROOT,queue_lock,atomic_json

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--stage',choices=['smoke','full'],required=True)
    p.add_argument('--output-base',required=True)
    p.add_argument('--detach',action='store_true')
    p.add_argument('--confirm-full')
    a=p.parse_args();base=Path(a.output_base).resolve()
    if a.stage=='full' and a.confirm_full!='SMOKE_REVIEWED_BY_USER':
        p.error('Full training requires explicit user confirmation after smoke review.')
    base.mkdir(parents=True,exist_ok=True)
    if a.detach:
        args=[sys.executable,'-u','-m','experiments.t11_core_confirmation.launch','--stage',a.stage,'--output-base',str(base)]
        if a.confirm_full:args+=['--confirm-full',a.confirm_full]
        name=f'launcher_{a.stage}_{time.time_ns()}'
        options=dict(cwd=ROOT,stdin=subprocess.DEVNULL)
        if os.name=='nt':options['creationflags']=subprocess.DETACHED_PROCESS|subprocess.CREATE_NEW_PROCESS_GROUP
        else:options['start_new_session']=True
        with (base/(name+'.log')).open('x',encoding='utf8') as log:
            child=subprocess.Popen(args,stdout=log,stderr=subprocess.STDOUT,**options)
        atomic_json(base/(name+'.json'),dict(pid=child.pid,command=args,log=name+'.log',state='launched_not_yet_passed'))
        print(json.dumps(dict(pid=child.pid,log=str(base/(name+'.log')),status=str(base/f'{a.stage}_supervisor.json'))));return
    # Separate supervisor lock covers the gaps between preflight and training.
    # A duplicate cannot write shared status or logs after failing this lock.
    with queue_lock(ROOT/'results/.t11_supervisor.lock'):
        status=base/f'{a.stage}_supervisor.json'
        def run(module,args):
            command=[sys.executable,'-u','-m',module,*map(str,args)]
            child=subprocess.Popen(command,cwd=ROOT)
            atomic_json(status,dict(state='running',stage=a.stage,pid=os.getpid(),child_pid=child.pid,command=command))
            returncode=child.wait()
            with (base/'child_exits.jsonl').open('a',encoding='utf8') as f:
                f.write(json.dumps(dict(command=command,pid=child.pid,returncode=returncode,time=time.time()))+'\n')
            if returncode!=0:raise RuntimeError(f'Child failed: {module}; exit code {returncode}')
        try:
            pre=base/'preflight'
            if a.stage=='smoke':
                run('experiments.t11_core_confirmation.readiness',['--output',base/'readiness.json'])
                if not pre.exists():run('experiments.t11_core_confirmation.checks',['--output',pre])
                else:
                    assert (pre/'preflight.json').exists(),'Failed/incomplete preflight: preserve and use a fresh attempt'
            args=['--stage',a.stage,'--output',base/a.stage,'--preflight',pre/'preflight.json']
            if a.stage=='full':args+=['--smoke-receipt',base/'smoke/receipt.json','--confirm-full',a.confirm_full]
            run('experiments.t11_core_confirmation.run',args)
            receipt=json.loads((base/a.stage/'receipt.json').read_text())
            assert receipt['passed']
            atomic_json(status,dict(state='complete',stage=a.stage,receipt=str(base/a.stage/'receipt.json'),full_authorized=False if a.stage=='smoke' else True))
            print('COMPLETE. Smoke does not authorize full training.',flush=True)
        except BaseException:
            atomic_json(status,dict(state='failed',stage=a.stage,error=traceback.format_exc()));raise

if __name__=='__main__':main()
