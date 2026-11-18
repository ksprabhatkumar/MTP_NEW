"""Portable T11 configuration, provenance, locks, and artifact helpers."""
import contextlib
import hashlib
import json
import os
import platform
import sys
from pathlib import Path
import torch

ROOT = Path(__file__).resolve().parents[2]
PACKAGE = Path(__file__).resolve().parent
MATRIX = json.loads((ROOT/'docs/experiments/t11_core_confirmation_r1_matrix.json').read_text())

def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(8*1024*1024), b''): h.update(block)
    return h.hexdigest()

def atomic_json(path, value):
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_name(path.name+'.tmp')
    tmp.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n',encoding='utf8')
    os.replace(tmp,path)

def atomic_torch(path,value):
    path=Path(path); tmp=path.with_name(path.name+'.tmp');torch.save(value,tmp);os.replace(tmp,path)

@contextlib.contextmanager
def queue_lock(path):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    f=path.open('a+b'); acquired=False
    try:
        if os.name=='nt':
            import msvcrt
            f.seek(0,2)
            if f.tell()==0:f.write(b'0');f.flush()
            f.seek(0);msvcrt.locking(f.fileno(),msvcrt.LK_NBLCK,1)
        else:
            import fcntl
            fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB)
        acquired=True
        yield
    finally:
        if acquired:
            if os.name=='nt':
                f.seek(0);msvcrt.locking(f.fileno(),msvcrt.LK_UNLCK,1)
            else:fcntl.flock(f,fcntl.LOCK_UN)
        f.close()

def inventory():
    # Include transitive project modules, not just the entry point.
    paths=list(ROOT.glob('*.py'))
    for directory in ('models','experiments/t08_predictive_20261114','experiments/t09_selective_ssm','experiments/t10_temporal','experiments/t11_core_confirmation'):
        paths.extend((ROOT/directory).rglob('*.py'))
    paths.extend((PACKAGE/'reference').glob('*'))
    paths.extend([PACKAGE/'dataset_manifest.json',PACKAGE/'reference_payload.json',ROOT/'docs/experiments/t11_core_confirmation_r1_matrix.json', ROOT/'requirements_workstation.txt'])
    return {p.relative_to(ROOT).as_posix():sha(p) for p in sorted(set(paths)) if p.is_file()}

def data_hashes():
    expected=json.loads((PACKAGE/'dataset_manifest.json').read_text())
    actual={name:sha(ROOT/name) for name in expected}
    assert actual==expected,'Dataset differs from audited transfer manifest; stop'
    return actual

def configure():
    assert torch.cuda.is_available(),'CUDA unavailable: do not silently train on CPU'
    assert torch.__version__.split('+')[0]=='2.5.1','Use the pinned environment'
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    torch.backends.cudnn.benchmark=False

def environment():
    return dict(python=sys.version,platform=platform.platform(),torch=torch.__version__,cuda=torch.version.cuda,
                gpu=torch.cuda.get_device_name(),vram_bytes=torch.cuda.get_device_properties(0).total_memory,
                precision='fp32',threads=torch.get_num_threads())

def fingerprint():
    return dict(sources=inventory(),data=data_hashes(),environment=environment())

def selected(stage):
    jobs=[dict(j,owner='workstation') for j in MATRIX['jobs']]
    if stage=='smoke':jobs=[j for j in jobs if j['subject'] in (2,15) and j['seed']==13]
    assert len(jobs)==(20 if stage=='smoke' else 90)
    assert len({j['id'] for j in jobs})==len(jobs)
    return jobs

class Tee:
    def __init__(self,console,log):self.console=console;self.log=log
    def write(self,value):self.console.write(value);self.log.write(value);self.log.flush()
    def flush(self):self.console.flush();self.log.flush()

@contextlib.contextmanager
def console_log(log):
    """Restore process streams before the enclosing log file is closed."""
    stdout,stderr=sys.stdout,sys.stderr
    sys.stdout,sys.stderr=Tee(stdout,log),Tee(stderr,log)
    try:
        yield
    finally:
        # Restore even on a training exception or log flush failure. Never
        # leave interpreter shutdown attempting to flush a closed log.
        sys.stdout,sys.stderr=stdout,stderr
        log.flush()
