"""Bounded recovery-worker diagnostics; no scientific queue or source edits."""
import argparse,faulthandler,hashlib,json,os,shutil,subprocess,sys,time,traceback
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
CASE='magnitude__ssm__none_clean__subject_15__seed_13'

def write(path,value):path.write_text(json.dumps(value,indent=2)+'\n',encoding='utf8')
def digest(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
    return h.hexdigest()

def worker(checkpoint):
    faulthandler.enable(all_threads=True)
    print('STAGE import_start',flush=True)
    sys.path.insert(0,str(ROOT))
    from experiments.t12_representation import checks
    print('STAGE import_complete',flush=True)
    # Trace only function boundaries and the original worker's own lines.
    # The original worker, precision, RNG restoration and optimizer step are
    # unchanged. Tracing is diagnostic instrumentation, not a new passing gate.
    watched={checks.worker.__code__,checks.objects.__code__,checks.step.__code__}
    def trace(frame,event,arg):
        if frame.f_code in watched and (event in ('call','return','exception') or (frame.f_code==checks.worker.__code__ and event=='line')):
            print(f'STAGE {frame.f_code.co_name} {event} line={frame.f_lineno}',flush=True)
        return trace if frame.f_code in watched else None
    sys.settrace(trace)
    try:checks.worker(str(checkpoint))
    finally:sys.settrace(None)
    print('STAGE worker_returned; awaiting normal process exit',flush=True)

def main():
    p=argparse.ArgumentParser();p.add_argument('--output');p.add_argument('--worker');a=p.parse_args()
    if a.worker:return worker(Path(a.worker))
    assert a.output
    out=Path(a.output).resolve();out.mkdir(parents=True,exist_ok=False)
    report=dict(scientific_training_launched=False,smoke_authorized=False,full_authorized=False)
    original=ROOT/'results/t12_representation_workstation_r2/magnitude/preflight'/(CASE+'.pt')
    try:
        assert original.is_file(),'Original failed-case checkpoint required from workstation; do not synthesize or retrain it'
        sys.path.insert(0,str(ROOT))
        from experiments.t12_representation.common import queue_lock,inventory,data_hashes
        # Keep the passing raw fingerprint reusable: this script lives outside
        # fingerprinted source directories. Require unchanged source and data.
        raw=json.loads((ROOT/'results/t12_representation_workstation_r2/raw/smoke/receipt.json').read_text())
        assert raw['passed'] and raw['jobs_passed']==20
        assert inventory()==raw['fingerprint']['sources'],'Workstation source changed: stop; do not edit hashes'
        assert data_hashes()==raw['fingerprint']['data']
        report['original_checkpoint_sha256']=digest(original)
        report['raw_smoke_preserved']=True
        with queue_lock(ROOT/'results/.t12_batch.lock'),queue_lock(ROOT/'results/.t11_supervisor.lock'),queue_lock(ROOT/'results/.t11_device.lock'):
            local=out/(CASE+'.pt');shutil.copyfile(original,local)
            assert digest(local)==report['original_checkpoint_sha256']
            previous=original.with_suffix('.actual.pt')
            report['original_worker_output_exists']=previous.exists()
            if previous.exists():report['original_worker_output_sha256']=digest(previous)
            env=os.environ.copy();env['MTP_T12_REPRESENTATION']='magnitude';env['PYTHONFAULTHANDLER']='1'
            command=[sys.executable,'-X','faulthandler','-u',str(Path(__file__).resolve()),'--worker',str(local)]
            report['command']=command;report['environment_overrides']={'MTP_T12_REPRESENTATION':'magnitude','PYTHONFAULTHANDLER':'1'}
            report['caveat']='Isolated diagnostic worker; original parent held corpus/GPU allocations. A successful replay is not a repaired preflight.'
            write(out/'diagnostic.json',dict(report,state='running'))
            with (out/'worker.log').open('w',encoding='utf8') as log:
                try:r=subprocess.run(command,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,timeout=300)
                except subprocess.TimeoutExpired:
                    report['worker_timeout_seconds']=300;raise
            report['returncode']=r.returncode;report['returncode_hex']=f'0x{r.returncode & 0xffffffff:08X}'
            report['worker_output_exists']=local.with_suffix('.actual.pt').exists()
            report['state']='diagnostic_completed' if r.returncode==0 else 'native_or_python_failure_reproduced'
            if r.returncode==0:
                assert report['worker_output_exists'],'Worker exited zero without output'
                # If the failed worker saved its step before shutdown, compare
                # it using the unchanged numerical comparator. Not a substitute
                # for preflight's unsaved in-parent expected step.
                if previous.exists():
                    import torch
                    from experiments.t12_representation.checks import compare
                    compare(torch.load(previous,map_location='cpu',weights_only=False),torch.load(local.with_suffix('.actual.pt'),map_location='cpu',weights_only=False))
                    report['original_worker_output_comparison']='passed at original tolerance; not parent-reference equivalence'
            assert digest(original)==report['original_checkpoint_sha256']
        # Optional native application crash records. No WMI, registry changes,
        # driver changes, dump settings or automatic retries.
        if os.name=='nt':
            cmd=['wevtutil','qe','Application','/q:*[System[(EventID=1000 or EventID=1001)]]','/c:12','/rd:true','/f:text']
            try:
                event=subprocess.run(cmd,capture_output=True,text=True,errors='replace',timeout=20)
                (out/'windows_application_events.txt').write_text(event.stdout+'\nSTDERR:\n'+event.stderr,encoding='utf8')
                report['event_query_returncode']=event.returncode
            except (OSError,subprocess.TimeoutExpired) as exc:report['event_query_error']=repr(exc)
        write(out/'diagnostic.json',report)
        print(json.dumps(report,indent=2))
        if r.returncode:raise SystemExit(1)
    except BaseException:
        report['error']=traceback.format_exc();write(out/'diagnostic.json',report);raise

if __name__=='__main__':main()
