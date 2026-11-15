"""Explicit independent RNG streams, unit-free raw stress, EMA repair losses."""
import copy
import torch
from torch.nn import functional as F
from o2_features import features
from run_local_message_pilot import precision_context

def streams(seed):
    return {name:torch.Generator().manual_seed(seed+offset) for name,offset in [('order',10000),('corruption',20000)]}

def corrupt_raw(raw,available,scale,generator):
    # Draw on CPU with a dedicated generator; model allocation/dropout cannot change stress.
    b,t,n,c=raw.shape
    rand=lambda *shape:torch.rand(shape,generator=generator)
    active=rand(b)<.75
    count=torch.randint(1,8,(b,),generator=generator)
    rank=rand(b,n).argsort(-1).argsort(-1)
    mask=(rank<count[:,None])&active[:,None]&available.cpu()
    gain=(.6+.8*rand(b,1,n,2)).repeat_interleave(3,-1).to(raw.device)
    knots=(2*rand(b,n,c,4)-1).reshape(b*n*c,1,4).to(raw.device)
    drift=F.interpolate(knots,size=t,mode='linear',align_corners=True).reshape(b,n,c,t).permute(0,3,1,2)
    drift=drift*.30*scale[None,None].to(raw.device)
    selected=mask.to(raw.device)
    changed=torch.where(selected[:,None,:,None],raw*gain+drift,raw)
    return changed,selected

def make_teacher(student):
    teacher=copy.deepcopy(student).eval()
    for p in teacher.parameters():p.requires_grad_(False)
    return teacher

@torch.no_grad()
def update_teacher(teacher,student,decay=.99):
    for a,b in zip(teacher.parameters(),student.parameters()):a.mul_(decay).add_(b,alpha=1-decay)
    for a,b in zip(teacher.buffers(),student.buffers()):
        if a.is_floating_point():a.mul_(decay).add_(b,alpha=1-decay)
        else:a.copy_(b)

def loss_for_batch(model,teacher,batch,fold,generator,epoch):
    with precision_context():
        clean,_,dc=model(batch['x'],batch['a'],return_diagnostics=True)
        ce_clean=F.cross_entropy(clean,batch['y'])
    zero=ce_clean.new_zeros(())
    if model.arm=='none_clean' or epoch<=2:
        return ce_clean,dict(ce_clean=ce_clean.detach(),ce_corrupt=zero,consistency=zero,repair=zero,quality=zero),clean
    perturbed,mask=corrupt_raw(batch['raw'],batch['a'],fold['raw_scale'],generator)
    with torch.no_grad():
        encoded=features(perturbed.float(),'mag_dot')
        encoded=(encoded-fold['mean'].to(encoded.device))/fold['std'].to(encoded.device)
        encoded=encoded*batch['a'][:,None,:,None]
        with precision_context():
            target,_,dt=teacher(batch['x'],batch['a'],return_diagnostics=True)
    with precision_context():
        changed,_,ds=model(encoded,batch['a'],return_diagnostics=True)
        ce_corrupt=F.cross_entropy(changed,batch['y'])
    consistency=F.kl_div(F.log_softmax(changed.float()/2,-1),F.softmax(target.float()/2,-1),reduction='batchmean')*4
    target_local=dt['local'].float().detach()
    scale=target_local.square().mean(-1).clamp_min(.01)
    error=(ds['prediction'].float()-target_local).square().mean(-1)/scale
    repair=(error*mask[:,None]).sum()/(mask.sum()*error.shape[1]).clamp_min(1)
    quality_target=torch.where(mask,torch.full_like(mask,.05,dtype=torch.float32),torch.ones_like(mask,dtype=torch.float32))
    qc=F.binary_cross_entropy_with_logits(dc['quality_logits'].float(),torch.ones_like(quality_target),reduction='none')
    qs=F.binary_cross_entropy_with_logits(ds['quality_logits'].float(),quality_target,reduction='none')
    quality=((qc+qs)*.5*batch['a']).sum()/batch['a'].sum()
    ramp=min(1.,(epoch-2)/3.)
    loss=.5*(ce_clean+ce_corrupt)+ramp*(.5*consistency+.1*repair+.1*quality)
    return loss,dict(ce_clean=ce_clean.detach(),ce_corrupt=ce_corrupt.detach(),consistency=consistency.detach(),repair=repair.detach(),quality=quality.detach()),clean
