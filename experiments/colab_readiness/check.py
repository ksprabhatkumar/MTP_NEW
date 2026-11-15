"""Read-only Colab readiness inventory, writing a fresh persistent report. No training."""
import argparse,hashlib,json,platform,shutil,sys
from pathlib import Path
import torch

def main():
    p=argparse.ArgumentParser();p.add_argument('--output',required=True);p.add_argument('--t09',required=True);a=p.parse_args()
    root=Path.cwd();drive=Path('/content/drive/MyDrive/mtp');out=Path(a.output)
    assert drive.is_dir(),'Mount Drive first'
    assert out.resolve().is_relative_to((drive/'results').resolve()),'Output must be persistent'
    out.mkdir(parents=True,exist_ok=False)
    report=dict(root=str(root),python=sys.version,platform=platform.platform(),torch=torch.__version__,cuda=torch.cuda.is_available(),gpu=torch.cuda.get_device_name() if torch.cuda.is_available() else None,total_gpu_bytes=torch.cuda.get_device_properties(0).total_memory if torch.cuda.is_available() else None,disk_free=shutil.disk_usage(drive).free,training_authorized=False)
    files=['models/selective_mamba_temporal.py','models/summer_dynamic.py','models/mamba_whar.py','experiments/t09_selective_ssm/check.py','run_o1_selection.py']
    report['current_sources']={f:hashlib.sha256((root/f).read_bytes()).hexdigest() for f in files}
    t09=Path(a.t09);report['t09']={}
    for f in ['status.json','audit.json','validation.json','calibration.json','reference.json','sources.json','environment.json']:
        path=t09/f;report['t09'][f]=json.loads(path.read_text()) if path.exists() else {'missing':True}
    report['source_mismatches']=[]
    for name,expected in report['t09']['sources.json'].items():
        if name=='missing':continue
        path=root/name.replace('\\','/')
        if not path.exists() or hashlib.sha256(path.read_bytes()).hexdigest()!=expected:report['source_mismatches'].append(name)
    report['cache']={}
    for label,path in [('local',root/'dataset/workstation_o1'),('persistent',drive/'dataset/workstation_o1')]:
        report['cache'][label]=dict(path=str(path),audit=(path/'audit.json').exists(),npz_count=len(list(path.glob('*.npz'))),hashes={f.name:hashlib.sha256(f.read_bytes()).hexdigest() for f in sorted(path.glob('*')) if f.suffix in ('.json','.npz')})
    report['cache_identical']=bool(report['cache']['local']['hashes']) and report['cache']['local']['hashes']==report['cache']['persistent']['hashes']
    (out/'readiness.json').write_text(json.dumps(report,indent=2));print(json.dumps({k:v for k,v in report.items() if k not in ('t09','cache')},indent=2))
if __name__=='__main__':main()
