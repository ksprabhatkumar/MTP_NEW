"""Revision22: one gated magnitude recovery, then authorized full queues."""
import argparse,json,os,subprocess,sys,time,traceback
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from experiments.t12_representation.common import configure,fingerprint,selected,queue_lock,atomic_json

def checked_child(command,env,log,record):
    child=subprocess.Popen(command,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT)
    record['child_pid']=child.pid;record['command']=command
    code=child.wait();record['returncode']=code
    record['returncode_hex']=f'0x{code & 0xffffffff:08X}'
    if code:raise RuntimeError(f'Child failed: {code} ({record["returncode_hex"]}); no retry')

def validate_smoke(base):
    receipt=json.loads((base/'smoke/receipt.json').read_text())
    pre=json.loads((base/'preflight/preflight.json').read_text())
    audit=json.loads((base/'smoke/audit.json').read_text())
    assert receipt['passed'] and receipt['stage']=='smoke' and receipt['jobs_passed']==20
    assert pre['passed'] and pre['cases']==20
    assert audit['passed'] and audit['audited_jobs']==20 and audit['paired_contrasts']==36
    assert receipt['fingerprint']==pre['fingerprint']==fingerprint(),'Original workstation/source/data fingerprint required'
    assert set(receipt['job_ids'])=={j['id'] for j in selected('smoke')}

def main():
    p=argparse.ArgumentParser();p.add_argument('--output',required=True)
    p.add_argument('--detach',action='store_true');p.add_argument('--authorize-full',action='store_true')
    a=p.parse_args();assert a.authorize_full,'Explicit user authorization for gated full required'
    out=Path(a.output).resolve();out.mkdir(parents=True,exist_ok=True)
    if a.detach:
        command=[sys.executable,'-u',str(Path(__file__).resolve()),'--output',str(out),'--authorize-full']
        options=dict(cwd=ROOT,stdin=subprocess.DEVNULL)
        if os.name=='nt':options['creationflags']=subprocess.DETACHED_PROCESS|subprocess.CREATE_NEW_PROCESS_GROUP
        else:options['start_new_session']=True
        log=out/f'launcher_{time.time_ns()}.log'
        with log.open('x',encoding='utf8') as f:child=subprocess.Popen(command,stdout=f,stderr=subprocess.STDOUT,**options)
        atomic_json(log.with_suffix('.json'),dict(pid=child.pid,command=command,log=str(log)))
        print(json.dumps(dict(pid=child.pid,log=str(log))));return
    with queue_lock(ROOT/'results/.t12_batch.lock'),queue_lock(ROOT/'results/.t11_supervisor.lock'):
        assert not (out/'status.json').exists(),'Attempt already exists: inspect; never silently retry'
        state=dict(state='running',pid=os.getpid(),completed=[],full_started=False)
        atomic_json(out/'status.json',state)
        try:
            configure();os.environ['MTP_T12_REPRESENTATION']='raw'
            raw=ROOT/'results/t12_representation_workstation_r2/raw';validate_smoke(raw)
            diag=json.loads((ROOT/'results/t12_native_diagnostics_r21/diagnostic.json').read_text())
            assert diag['returncode']==0 and diag['worker_output_exists']
            magnitude=out/'magnitude';assert not magnitude.exists() and not (out/'raw/full').exists()
            def run(label,module,args,rep):
                env=os.environ.copy();env['MTP_T12_REPRESENTATION']=rep
                env['PYTHONFAULTHANDLER']='1';env['PYTHONUNBUFFERED']='1'
                command=[sys.executable,'-X','faulthandler','-u','-m',module,*map(str,args)]
                state.update(stage=label,representation=rep);atomic_json(out/'status.json',state)
                rec=dict(stage=label,representation=rep,command=command,started=time.time())
                with (out/(label+'.log')).open('x',encoding='utf8') as log:
                    try:checked_child(command,env,log,rec)
                    finally:atomic_json(out/(label+'_exit.json'),rec)
                state['completed'].append(label);atomic_json(out/'status.json',state)
            run('magnitude_readiness','experiments.t12_representation.readiness',['--output',magnitude/'readiness.json'],'magnitude')
            run('magnitude_preflight','experiments.t12_representation.checks',['--output',magnitude/'preflight'],'magnitude')
            run('magnitude_smoke','experiments.t12_representation.run',['--stage','smoke','--output',magnitude/'smoke','--preflight',magnitude/'preflight/preflight.json'],'magnitude')
            # Independently require BOTH receipts before starting full.
            for rep,base in [('raw',raw),('magnitude',magnitude)]:
                os.environ['MTP_T12_REPRESENTATION']=rep;validate_smoke(base)
            atomic_json(out/'full_gate.json',dict(passed=True,user_authorized=True,raw_smoke=str(raw/'smoke/receipt.json'),magnitude_smoke=str(magnitude/'smoke/receipt.json'),native_cause_resolved=False))
            for rep,base in [('raw',raw),('magnitude',magnitude)]:
                state['full_started']=True
                run(rep+'_full','experiments.t12_representation.run',['--stage','full','--output',out/rep/'full','--preflight',base/'preflight/preflight.json','--smoke-receipt',base/'smoke/receipt.json','--confirm-full','SMOKE_REVIEWED_BY_USER'],rep)
                receipt=json.loads((out/rep/'full/receipt.json').read_text())
                target=json.loads((out/rep/'full/target_evaluation/audit.json').read_text())
                assert receipt['passed'] and receipt['jobs_passed']==90 and target['passed'] and target['primary_rows']==270
            state['state']='complete';atomic_json(out/'status.json',state)
        except BaseException:
            state.update(state='failed',error=traceback.format_exc());atomic_json(out/'status.json',state);raise

if __name__=='__main__':main()
