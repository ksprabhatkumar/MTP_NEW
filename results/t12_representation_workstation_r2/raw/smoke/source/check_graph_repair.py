"""Executable correctness and recovery checks before the visible training pilot."""
import argparse,copy,hashlib,json
from contextlib import nullcontext
from pathlib import Path
import torch
import graph_repair_training as training
from models.summer_graph_repair import ARMS,SummerGraphRepair
from models.summer_dynamic import SummerDynamicWHAR
from o2_features import features
from run_o1_selection import rng_state,restore_rng
from run_graph_repair import ROOT,SOURCES
from workstation_experiments import seed_all,save_checkpoint,write_json

def main():
    p=argparse.ArgumentParser();p.add_argument('--output',required=True);args=p.parse_args()
    out=Path(args.output);out.mkdir(parents=True,exist_ok=True)
    assert torch.cuda.is_available();torch.set_num_threads(4)
    checks=[];seed_all(13)
    x=torch.randn(8,50,9,5,device='cuda');a=torch.ones(8,9,dtype=torch.bool,device='cuda')
    seed_all(13);base=SummerDynamicWHAR(in_features=5,graph_variant='none').cuda().eval()
    with torch.no_grad():reference=base(x,a)[0]
    for arm in ARMS:
        seed_all(13);m=SummerGraphRepair(arm).cuda().eval()
        with torch.no_grad():torch.testing.assert_close(m(x,a)[0],reference,rtol=0,atol=0)
    checks.append('All four arms initialize to the established no-graph output')
    del base
    with torch.no_grad():
        m.repair_output.weight.copy_(torch.eye(16,device='cuda'))
        _,_,d=m(x,a,True)
        torch.testing.assert_close(d['adjacency'].sum(-1)+d['null_mass'],torch.ones_like(d['null_mass']))
        assert (d['adjacency'].diagonal(dim1=-2,dim2=-1)==0).all()
        m.message_intervention='zero_message';_,_,z=m(x,a,True);assert (z['correction']==0).all();m.message_intervention=None
        a[:,4]=False;altered=x.clone();altered[:,:,4]=10000
        torch.testing.assert_close(m(x,a)[0],m(altered,a)[0],rtol=0,atol=0)
        _,_,d=m(x,a,True);assert (d['adjacency'][:,:,4]==0).all()
        a[:]=False;a[:,0]=True;_,_,d=m(x,a,True);assert (d['null_mass'][:,0]==1).all()
        a[:]=True;_,_,d=m(x,a,True);m.quality[-1].bias.fill_(-30);_,_,low=m(x,a,True)
        assert (low['null_mass']>d['null_mass']).all()
        local=SummerGraphRepair('local_repair').cuda().eval();local.repair_output.weight.copy_(torch.eye(16,device='cuda'))
        _,_,before=local(x,a,True);altered=x.clone();altered[:,:,1:]+=10;_,_,after=local(altered,a,True)
        torch.testing.assert_close(before['correction'][:,:,0],after['correction'][:,:,0],rtol=0,atol=0)
    checks.append('Null probability, no self edges, missing-sender masking, zero-sender cancellation, and local control isolation')
    del m,local
    # Recovery test uses FP32 to distinguish recovery failures from autocast rounding.
    training.precision_context=lambda:nullcontext()
    raw=torch.randn(8,50,9,6,device='cuda');encoded=features(raw,'mag_dot')
    mean=encoded.mean((0,1),keepdim=True);std=encoded.std((0,1),keepdim=True).clamp_min(1e-6)
    batch=dict(raw=raw,x=(encoded-mean)/std,a=a,y=torch.arange(8,device='cuda'))
    fold=dict(mean=mean,std=std,raw_scale=torch.ones(9,6))
    original=raw.clone();g=training.streams(13)['corruption'];changed,mask=training.corrupt_raw(raw,a,fold['raw_scale'],g)
    assert (mask.sum(-1)<=3).all();torch.testing.assert_close(raw,original,rtol=0,atol=0)
    assert (changed[~mask[:,None,:,None].expand_as(raw)]==raw[~mask[:,None,:,None].expand_as(raw)]).all()
    seed_all(13);m=SummerGraphRepair('graph_repair').cuda().train();teacher=training.make_teacher(m)
    opt=torch.optim.Adam(m.parameters(),lr=1e-4,fused=True);scheduler=torch.optim.lr_scheduler.StepLR(opt,1,.99);gens=training.streams(13)
    def step():
        opt.zero_grad(set_to_none=True)
        loss,parts,_=training.loss_for_batch(m,teacher,batch,fold,gens['corruption'],3)
        assert torch.isfinite(loss) and all(torch.isfinite(v) for v in parts.values())
        loss.backward();torch.nn.utils.clip_grad_norm_(m.parameters(),1.,error_if_nonfinite=True)
        opt.step();scheduler.step();training.update_teacher(teacher,m);return loss.detach().clone()
    step();step()
    assert m.relation_value[0].weight.grad.abs().sum()>0
    assert all(p.grad is None for p in teacher.parameters())
    checkpoint=dict(model=m.state_dict(),teacher=teacher.state_dict(),optimizer=opt.state_dict(),scheduler=scheduler.state_dict(),streams={k:v.get_state() for k,v in gens.items()},**rng_state())
    save_checkpoint(out/'recovery_check.pt',checkpoint)
    expected_loss=step();expected={k:v.clone() for k,v in m.state_dict().items()};expected_teacher={k:v.clone() for k,v in teacher.state_dict().items()}
    state=torch.load(out/'recovery_check.pt',map_location='cpu',weights_only=False)
    m.load_state_dict(state['model']);teacher.load_state_dict(state['teacher']);opt.load_state_dict(state['optimizer']);scheduler.load_state_dict(state['scheduler'])
    restore_rng(state)
    for k,v in gens.items():v.set_state(state['streams'][k])
    restored_loss=step();torch.testing.assert_close(expected_loss,restored_loss,rtol=0,atol=1e-6)
    for k,v in m.state_dict().items():torch.testing.assert_close(v,expected[k],rtol=0,atol=1e-6)
    for k,v in teacher.state_dict().items():torch.testing.assert_close(v,expected_teacher[k],rtol=0,atol=1e-6)
    checks.append('Finite augmented losses/gradients, frozen teacher, and exact optimizer/scheduler/EMA/RNG recovery within 1e-6')
    write_json(out/'preflight.json',dict(passed=True,checks=checks,source_hashes={n:hashlib.sha256((ROOT/n).read_bytes()).hexdigest() for n in SOURCES},gpu=torch.cuda.get_device_name(),peak_memory_mb=torch.cuda.max_memory_allocated()/1024**2))
    print('PREFLIGHT PASSED: '+ '; '.join(checks),flush=True)

if __name__=='__main__':main()
