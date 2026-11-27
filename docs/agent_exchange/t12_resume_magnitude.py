"""Revision22: one controlled magnitude-only recovery attempt; smoke only."""
import argparse,json,os,subprocess,sys,time,traceback
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]

def main():
    p=argparse.ArgumentParser();p.add_argument('--output-root',required=True);p.add_argument('--detach',action='store_true');a=p.parse_args()
    out=Path(a.output_root).resolve()
    if a.detach:
        out.mkdir(parents=True,exist_ok=False)
        log=out/'recovery_supervisor.log'
        args=[sys.executable,'-X','faulthandler','-u',str(Path(__file__).resolve()),'--output-root',str(out)]
        options=dict(cwd=ROOT,stdin=subprocess.DEVNULL)
        if os.name=='nt':options['creationflags']=subprocess.DETACHED_PROCESS|subprocess.CREATE_NEW_PROCESS_GROUP
        else:options['start_new_session']=True
        env=os.environ.copy();env['PYTHONFAULTHANDLER']='1'
        with log.open('x',encoding='utf8') as f:child=subprocess.Popen(args,stdout=f,stderr=subprocess.STDOUT,env=env,**options)
        (out/'launch.json').write_text(json.dumps(dict(pid=child.pid,command=args,log=str(log)),indent=2),encoding='utf8')
        print(json.dumps(dict(pid=child.pid,log=str(log))));return
    out.mkdir(parents=True,exist_ok=True)
    sys.path.insert(0,str(ROOT))
    from experiments.t12_representation.common import inventory,data_hashes,queue_lock,atomic_json
    with queue_lock(ROOT/'results/.t12_batch.lock'):
        status=out/'recovery_status.json'
        try:
            assert not (out/'magnitude').exists(),'Existing recovery attempt; no automatic retry'
            rawbase=ROOT/'results/t12_representation_workstation_r2/raw'
            raw=json.loads((rawbase/'smoke/receipt.json').read_text())
            assert raw['passed'] and raw['jobs_passed']==20
            assert json.loads((rawbase/'smoke/audit.json').read_text())['passed']
            assert inventory()==raw['fingerprint']['sources'],'Source changed; do not relax fingerprint'
            assert data_hashes()==raw['fingerprint']['data'],'Dataset changed'
            diag=json.loads((ROOT/'results/t12_native_diagnostics_r21/diagnostic.json').read_text())
            assert diag['returncode']==0 and diag['worker_output_exists'] and diag['state']=='diagnostic_completed'
            env=os.environ.copy();env['PYTHONFAULTHANDLER']='1'
            args=[sys.executable,'-X','faulthandler','-u','-m','experiments.t12_representation.launch','--representation','magnitude','--stage','smoke','--output-base',str(out/'magnitude')]
            child=subprocess.Popen(args,cwd=ROOT,env=env)
            atomic_json(status,dict(state='running',pid=os.getpid(),child_pid=child.pid,command=args,raw_smoke_reused=str(rawbase/'smoke/receipt.json'),full_authorized=False))
            rc=child.wait()
            atomic_json(out/'child_exit.json',dict(returncode=rc,hex=f'0x{rc & 0xffffffff:08X}',command=args))
            assert rc==0,f'Magnitude attempt failed, exit {rc}; no retry'
            base=out/'magnitude';receipt=json.loads((base/'smoke/receipt.json').read_text());audit=json.loads((base/'smoke/audit.json').read_text())
            assert receipt['passed'] and receipt['jobs_passed']==20 and audit['passed'] and audit['paired_contrasts']==36
            pre=json.loads((base/'preflight/preflight.json').read_text());assert pre['passed'] and pre['cases']==20 and pre['fingerprint']==receipt['fingerprint']
            assert raw['fingerprint']['sources']==receipt['fingerprint']['sources'] and raw['fingerprint']['data']==receipt['fingerprint']['data']
            rawenv=dict(raw['fingerprint']['environment']);magenv=dict(receipt['fingerprint']['environment'])
            assert rawenv.pop('representation')=='raw' and magenv.pop('representation')=='magnitude' and rawenv==magenv
            atomic_json(out/'smoke_pair.json',dict(passed=True,raw_receipt=str(rawbase/'smoke/receipt.json'),magnitude_receipt=str(base/'smoke/receipt.json'),jobs=40,full_authorized=False))
            atomic_json(status,dict(state='complete',raw_rerun=False,magnitude_smoke_jobs=20,full_authorized=False))
        except BaseException:
            atomic_json(status,dict(state='failed',error=traceback.format_exc(),full_authorized=False));raise

if __name__=='__main__':main()
