"""Record the actual workstation and verify the transferred dataset."""
import argparse,importlib.metadata,json,platform,shutil,subprocess,sys,traceback
from pathlib import Path
from .common import ROOT,configure,environment,inventory,data_hashes,atomic_json

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
        # Keep raw inventory for driver, CPU/RAM and competing process review.
        commands=[['nvidia-smi']]
        if sys.platform=='win32':commands.append(['powershell','-NoProfile','-Command','Get-CimInstance Win32_ComputerSystem | Select-Object TotalPhysicalMemory,NumberOfLogicalProcessors | ConvertTo-Json'])
        else:commands.append(['sh','-c','cat /proc/meminfo; getconf _NPROCESSORS_ONLN'])
        record['hardware_inventory']=[]
        for command in commands:
            r=subprocess.run(command,capture_output=True,text=True,check=True)
            record['hardware_inventory'].append(dict(command=command,output=r.stdout))
        record['pip_freeze']=subprocess.check_output([sys.executable,'-m','pip','freeze'],text=True)
        record['passed']=True
    except BaseException:
        record['error']=traceback.format_exc();raise
    finally:atomic_json(out,record)
    print(json.dumps(dict(passed=True,gpu=record['environment']['gpu'],dataset_files=len(record['dataset']))))

if __name__=='__main__':main()
