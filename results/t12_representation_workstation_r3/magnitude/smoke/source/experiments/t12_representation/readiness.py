"""Record the actual workstation and verify the transferred dataset."""
import argparse,importlib.metadata,json,os,platform,shutil,subprocess,sys,traceback
from pathlib import Path
from .common import ROOT,configure,environment,inventory,data_hashes,atomic_json

def system_resources():
    """Read actual RAM/CPU directly; Windows does not require PowerShell/WMI."""
    if sys.platform=='win32':
        import ctypes
        class MemoryStatus(ctypes.Structure):
            _fields_=[('length',ctypes.c_uint32),('load',ctypes.c_uint32)]+[(n,ctypes.c_uint64) for n in ('total_phys','avail_phys','total_page','avail_page','total_virtual','avail_virtual','avail_extended')]
        status=MemoryStatus();status.length=ctypes.sizeof(status)
        api=ctypes.WinDLL('kernel32',use_last_error=True).GlobalMemoryStatusEx
        api.argtypes=[ctypes.POINTER(MemoryStatus)];api.restype=ctypes.c_int
        if not api(ctypes.byref(status)):raise ctypes.WinError(ctypes.get_last_error())
        total=int(status.total_phys);source='Win32 GlobalMemoryStatusEx; os.cpu_count'
    else:
        total=os.sysconf('SC_PAGE_SIZE')*os.sysconf('SC_PHYS_PAGES');source='POSIX sysconf; os.cpu_count'
    count=os.cpu_count()
    if not isinstance(total,int) or total<=0 or not isinstance(count,int) or count<=0:
        raise RuntimeError('Unable to determine positive physical RAM / logical CPU count')
    return dict(total_physical_memory_bytes=total,logical_processors=count,source=source)

def collect_hardware(record):
    record['hardware_inventory']=[]
    command=['nvidia-smi']
    try:
        result=subprocess.run(command,capture_output=True,text=True,errors='replace',timeout=30,check=False)
    except (OSError,subprocess.TimeoutExpired) as exc:
        record['hardware_inventory'].append(dict(command=command,error=repr(exc)))
        raise
    record['hardware_inventory'].append(dict(command=command,output=result.stdout,stderr=result.stderr,returncode=result.returncode))
    if result.returncode:
        raise RuntimeError(f'nvidia-smi failed with exit {result.returncode}; see recorded stdout/stderr')
    record['system_resources']=system_resources()

def main():
    p=argparse.ArgumentParser();p.add_argument('--output',required=True);a=p.parse_args();out=Path(a.output)
    assert not out.exists(),'Preserve readiness: choose a fresh attempt directory'
    record=dict(passed=False)
    try:
        assert sys.version_info[:2]==(3,11),'Create Prabhat using Python 3.11'
        configure();record['environment']=environment()
        packages={}
        for line in (ROOT/'requirements_workstation.txt').read_text().splitlines():
            if not line or line.startswith('#'):continue
            name,version=line.split('==');actual=importlib.metadata.version(name)
            assert actual==version,(name,actual,version);packages[name]=actual
        record.update(packages=packages,sources=inventory(),dataset=data_hashes(),disk_free_bytes=shutil.disk_usage(ROOT).free)
        # Driver/process inventory remains required. RAM/CPU use native APIs.
        collect_hardware(record)
        record['pip_freeze']=subprocess.check_output([sys.executable,'-m','pip','freeze'],text=True)
        record['passed']=True
    except BaseException:
        record['error']=traceback.format_exc();raise
    finally:atomic_json(out,record)
    print(json.dumps(dict(passed=True,gpu=record['environment']['gpu'],dataset_files=len(record['dataset']))))

if __name__=='__main__':main()
