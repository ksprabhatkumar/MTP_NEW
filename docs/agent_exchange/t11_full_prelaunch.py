"""Read-only workstation gate for revision 18. No training and no artifact writes."""
import json
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
import torch
import numpy as np
from experiments.t11_core_confirmation.common import configure,fingerprint,selected,sha
from workstation_experiments import metric_values

def main():
    base=ROOT/'results/t11_workstation_r2';smoke=base/'smoke'
    read=lambda p:json.loads(p.read_text(encoding='utf8'))
    configure();fp=fingerprint()
    pre=read(base/'preflight/preflight.json');receipt=read(smoke/'receipt.json')
    assert fp==pre['fingerprint']==receipt['fingerprint']==read(smoke/'fingerprint.json'),'Fingerprint changed: stop, do not repair hashes'
    ready=read(base/'readiness.json')
    assert ready['passed'] and ready['sources']==fp['sources'] and ready['dataset']==fp['data'] and ready['environment']==fp['environment']
    assert pre['passed'] and pre['cases']==20
    assert receipt['passed'] and receipt['stage']=='smoke' and receipt['jobs_passed']==20
    assert set(receipt['job_ids'])=={j['id'] for j in selected('smoke')}
    audit=read(smoke/'audit.json');assert audit['passed'] and audit['audited_jobs']==20 and audit['paired_contrasts']==36 and audit==receipt['audit']
    assert read(base/'smoke_supervisor.json')['state']=='complete'
    exits=[json.loads(x) for x in (base/'child_exits.jsonl').read_text().splitlines() if x.strip()]
    assert len(exits)==3 and all(x['returncode']==0 for x in exits)
    assert not (base/'full').exists(),'Full output already exists: inspect, do not launch duplicate'
    for name,h in fp['sources'].items():
        assert sha(smoke/'source'/name)==h,('Workstation source snapshot bytes changed',name)
    for job in selected('smoke'):
        folder=smoke/'jobs'/job['id'];result=read(folder/'result.json');cfg=result['config']
        latest=torch.load(folder/'latest.pt',map_location='cpu',weights_only=False)
        best=torch.load(folder/'best.pt',map_location='cpu',weights_only=False)
        norm=torch.load(smoke/f"normalization_{job['subject']:02d}.pt",map_location='cpu',weights_only=False)
        assert latest['config']==best['config']==cfg and cfg['epochs']==4
        assert latest['epoch']==4 and latest['history']==result['history'] and len(result['history'])==4
        assert best['epoch']==result['best_epoch']==max(result['history'],key=lambda x:x['val_ideal']['macro_f1'])['epoch']
        assert all(k in latest for k in ('model','teacher','optimizer','scheduler','streams','cpu_rng','cuda_rng','raw_scale'))
        for name in ('mean','std'):assert torch.equal(norm[name],latest[name]) and torch.equal(norm[name],best[name])
        for split in ('val_ideal','val_self'):
            with np.load(folder/f'predictions_{split}.npz') as d:
                assert np.array_equal(d['corpus_indices'],norm['indices'][split])
                assert not np.intersect1d(d['corpus_indices'],norm['indices']['train']).size
                assert np.isfinite(d['logits']).all() and np.array_equal(d['predictions'],d['logits'].argmax(-1))
                m=metric_values(d['labels'],d['predictions'])
                for key in ('accuracy','macro_f1','weighted_f1'):assert abs(m[key]-result['metrics'][split][key])<1e-10
    print('PASS: original workstation fingerprints, snapshots, 20 checkpoint pairs, normalization and 40 prediction artifacts; no writes or training.')

if __name__=='__main__':main()
