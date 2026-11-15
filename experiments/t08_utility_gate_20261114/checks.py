"""One-step continuation recovery for both phases, executed by Luna during smoke."""
import copy
from contextlib import nullcontext
import torch
from experiments.t08_utility_gate_20261114.model import set_stage
import experiments.t08_utility_gate_20261114.training as training
from run_o1_selection import rng_state,restore_rng
from workstation_experiments import save_checkpoint,write_json

def recovery(model,fold,job,config):
    initial=copy.deepcopy(model.state_dict());initial_rng=rng_state();old_context=training.precision_context
    batch={k:v[:8].cuda() for k,v in fold['data']['train'].items()}
    checks=[]
    old_matmul=torch.backends.cuda.matmul.allow_tf32;old_cudnn=torch.backends.cudnn.allow_tf32
    try:
        training.precision_context=lambda:nullcontext()
        torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
        for phase in ('correction','gate'):
            set_stage(model,phase);params=[p for p in model.parameters() if p.requires_grad]
            opt=torch.optim.Adam(params,lr=1e-4,weight_decay=5e-4);gens=training.streams(config['seed'])
            def step():
                opt.zero_grad(set_to_none=True);loss,parts=training.loss_batch(model,batch,fold,gens['corruption'],phase)
                loss.backward();assert any(p.grad is not None and p.grad.abs().sum()>0 for p in params)
                torch.nn.utils.clip_grad_norm_(params,1.,error_if_nonfinite=True);opt.step();return loss.detach().clone()
            step()
            checkpoint=job/f'recovery_{phase}.pt'
            save_checkpoint(checkpoint,dict(model=model.state_dict(),optimizer=opt.state_dict(),streams={k:g.get_state() for k,g in gens.items()},**rng_state()))
            expected_loss=step();expected=copy.deepcopy(model.state_dict())
            saved=torch.load(checkpoint,map_location='cpu',weights_only=False)
            model.load_state_dict(saved['model']);opt.load_state_dict(saved['optimizer']);restore_rng(saved)
            for k,g in gens.items():g.set_state(saved['streams'][k])
            actual=step();torch.testing.assert_close(expected_loss,actual,rtol=0,atol=1e-6)
            for k,v in model.state_dict().items():torch.testing.assert_close(v,expected[k],rtol=0,atol=1e-6)
            checks.append(phase+' optimizer/model/RNG continuation within 1e-6')
    finally:
        torch.backends.cuda.matmul.allow_tf32=old_matmul;torch.backends.cudnn.allow_tf32=old_cudnn
        training.precision_context=old_context;model.load_state_dict(initial);restore_rng(initial_rng);set_stage(model,'correction')
    write_json(job/'recovery.json',dict(passed=True,checks=checks))
