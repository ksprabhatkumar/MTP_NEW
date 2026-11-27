"""Read originals and save a NEW compact inventory before T12; no training."""
import argparse,hashlib,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
import torch
from experiments.t11_core_confirmation.common import sha

def main():
    p=argparse.ArgumentParser();p.add_argument('--output',required=True);a=p.parse_args()
    out=Path(a.output);assert not out.exists(),'Preserve prior inventory; use a fresh path'
    base=ROOT/'results/t11_workstation_r2/full'
    receipt=json.loads((base/'receipt.json').read_text());assert receipt['passed'] and receipt['jobs_passed']==90
    records=[]
    for job in receipt['job_ids']:
        d=base/'jobs'/job;result=json.loads((d/'result.json').read_text())
        for name in ('best.pt','latest.pt'):
            f=d/name;s=torch.load(f,map_location='cpu',weights_only=False)
            assert s['config']==result['config']
            assert s['epoch']==(31 if name=='latest.pt' else result['best_epoch'])
            if name=='latest.pt':
                assert len(s['history'])==31
                assert all(k in s for k in ('teacher','optimizer','scheduler','streams','cpu_rng','cuda_rng','mean','std','raw_scale'))
            records.append(dict(path=str(f.relative_to(ROOT)),bytes=f.stat().st_size,sha256=sha(f),epoch=s['epoch']))
    for subject in (2,5,15):
        f=base/f'normalization_{subject:02d}.pt';s=torch.load(f,map_location='cpu',weights_only=False)
        assert all(k in s for k in ('mean','std','indices'))
        records.append(dict(path=str(f.relative_to(ROOT)),bytes=f.stat().st_size,sha256=sha(f)))
    out.parent.mkdir(parents=True,exist_ok=True)
    out.write_text(json.dumps(dict(passed=True,checkpoint_pairs=90,normalization_files=3,records=records),indent=2)+'\n',encoding='utf8')
    print('PASS: 90 checkpoint pairs and 3 normalization files preserved and inventoried.')

if __name__=='__main__':main()
