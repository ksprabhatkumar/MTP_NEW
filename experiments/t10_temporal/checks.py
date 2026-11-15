"""T10 preflight: ten real-data FP32 backward paths, sharing and restart gate."""
import argparse,copy,json,random
from pathlib import Path
import numpy as np
import torch
from experiments.t10_temporal.model import ARMS,TEMPORALS,TemporalFactorial,optimizer_groups
from experiments.t10_temporal.training import streams,make_teacher,loss_for_batch,update_teacher
from run_o1_selection import build_fold,restore_rng,rng_state
from o2_features import O2Corpus
from workstation_experiments import seed_all
from experiments.t10_temporal.run import inventory,data_hashes
ROOT=Path(__file__).resolve().parents[2]
def main():
 p=argparse.ArgumentParser();p.add_argument('--output',required=True);p.add_argument('--owner',required=True);a=p.parse_args();out=Path(a.output);out.mkdir(parents=True,exist_ok=True)
 assert torch.cuda.is_available();torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
 corpus=O2Corpus();fold=build_fold(corpus,2);ix=fold['indices']['train'];raw=corpus.x[ix];fold['raw_scale']=raw.std((0,1)).reshape(9,2,3).square().mean(-1).sqrt().clamp_min(1e-4).repeat_interleave(3,-1);fold['data']['train']['raw']=raw
 batch={k:v[:8].cuda() for k,v in fold['data']['train'].items()};fold['raw_scale']=fold['raw_scale'];records=[];initial={}
 # Every arm/backbone has common shared initialization under identical seed.
 for temporal in TEMPORALS:
  for arm in ARMS:
   seed_all(13);m=TemporalFactorial(arm,temporal).cuda();key=(temporal,arm)
   probe=torch.randn(3,50,9,5,device='cuda');available=torch.ones(3,9,dtype=torch.bool,device='cuda');m.eval()
   with torch.no_grad():
    if arm=='graph_repair':
     _,_,before=m(probe,available,True);changed=probe.clone();changed[:,:,0]+=11;_,_,after=m(changed,available,True);torch.testing.assert_close(before['prediction'][:,:,0],after['prediction'][:,:,0],atol=1e-6,rtol=0);assert (before['prediction'][:,:,1:]-after['prediction'][:,:,1:]).abs().max()>0
    if arm=='local_repair':
     _,_,before=m(probe,available,True);changed=probe.clone();changed[:,:,1:]+=11;_,_,after=m(changed,available,True);torch.testing.assert_close(before['prediction'][:,:,0],after['prediction'][:,:,0],atol=0,rtol=0)
    missing=available.clone();missing[:,4]=False;altered=probe.clone();altered[:,:,4]=1e4;torch.testing.assert_close(m(altered,missing)[0],m(probe,missing)[0],atol=1e-6,rtol=0)
   m.train()
   shared={k:v.detach().cpu().clone() for k,v in m.state_dict().items() if not k.startswith('mamba.')};initial[key]=shared
   teacher=make_teacher(m);gens=streams(13);opt=torch.optim.Adam(optimizer_groups(m),lr=1e-4,betas=(.9,.999),eps=1e-8,fused=False,foreach=False)
   m.train();opt.zero_grad(set_to_none=True);loss,parts,logits=loss_for_batch(m,teacher,batch,fold,gens['corruption'],3)
   assert torch.isfinite(loss) and torch.isfinite(logits).all();loss.backward();torch.nn.utils.clip_grad_norm_(m.parameters(),1.,error_if_nonfinite=True)
   grads={n:float(p.grad.abs().sum()) for n,p in m.named_parameters() if p.grad is not None}
   if temporal=='ssm':
    for term in ('A_log','D','x_proj','dt_proj','conv1d'):
     assert any(term in n and v>0 for n,v in grads.items()),f'no SSM gradient: {term}'
   # Check that stochastic continuation survives full state/RNG serialization.
   opt.step();update_teacher(teacher,m);saved=dict(model=copy.deepcopy(m.state_dict()),teacher=copy.deepcopy(teacher.state_dict()),optimizer=copy.deepcopy(opt.state_dict()),streams={k:g.get_state() for k,g in gens.items()},rng=rng_state())
   x=batch['x'];av=batch['a'];m.eval();teacher.eval()
   with torch.no_grad():expected=m(x,av)[0].cpu()
   m.load_state_dict(saved['model']);teacher.load_state_dict(saved['teacher']);opt.load_state_dict(saved['optimizer']);restore_rng(saved['rng'])
   for k,g in gens.items():g.set_state(saved['streams'][k])
   with torch.no_grad():actual=m(x,av)[0].cpu()
   torch.testing.assert_close(actual,expected,atol=1e-6,rtol=0)
   records.append(dict(arm=arm,temporal=temporal,loss=float(loss),finite_gradients=True,ssm_gradient_checks=(temporal=='ssm'),checkpoint_replay=True,gradient_norms=grads));del m,teacher,opt;torch.cuda.empty_cache()
 for t in TEMPORALS:
  reference=initial[(t,ARMS[0])]
  for arm in ARMS[1:]:
   for k,v in reference.items():torch.testing.assert_close(v,initial[(t,arm)][k],atol=0,rtol=0,msg=f'initialization mismatch {t}/{arm}/{k}')
 payload=dict(passed=len(records)==10 and all(x['checkpoint_replay'] for x in records),owner=a.owner,jobs=len(records),records=records,device=torch.cuda.get_device_name(),sources=inventory(),data=data_hashes())
 (out/'preflight.json').write_text(json.dumps(payload,indent=2)+'\n');assert payload['passed'];print(json.dumps(payload))
if __name__=='__main__':main()
