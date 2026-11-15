"""Frozen-backbone correction learning and detached training-only utility targets."""
import torch
from torch.nn import functional as F
from o2_features import features
from run_local_message_pilot import precision_context
from experiments.t08_predictive_20261114.training import corrupt_raw,streams

def loss_batch(model,batch,fold,generator,phase):
    perturbed,mask=corrupt_raw(batch['raw'],batch['a'],fold['raw_scale'],generator)
    changed=features(perturbed.float(),'mag_dot')
    changed=(changed-fold['mean'].to(changed.device))/fold['std'].to(changed.device)
    changed*=batch['a'][:,None,:,None]
    with precision_context():
        _,_,clean=model(batch['x'],batch['a'],True)
        _,_,stress=model(changed,batch['a'],True)
    y=batch['y'];a=batch['a']
    parts={}
    if phase=='correction':
        base=F.cross_entropy(clean['p0'].float(),y,reduction='none').detach()
        ce=F.cross_entropy(clean['p1'].float(),y,reduction='none')
        stressed=F.cross_entropy(stress['p1'].float(),y)
        target=clean['local'].float().detach()
        error=(stress['prediction'].float()-target).square().mean(-1)/target.square().mean(-1).clamp_min(.01)
        prediction=(error*mask[:,None]).sum()/(mask.sum()*error.shape[1]).clamp_min(1)
        qt=torch.where(mask,torch.full_like(mask,.05,dtype=torch.float32),torch.ones_like(mask,dtype=torch.float32))
        qc=F.binary_cross_entropy_with_logits(clean['quality_logits'].float(),torch.ones_like(qt),reduction='none')
        qs=F.binary_cross_entropy_with_logits(stress['quality_logits'].float(),qt,reduction='none')
        quality=((qc+qs)*.5*a).sum()/a.sum()
        harm=F.relu(ce-base).mean()
        loss=.5*(ce.mean()+stressed)+.1*prediction+.1*quality+.5*harm
        parts=dict(ce_clean=ce.mean(),ce_corrupt=stressed,prediction=prediction,quality=quality,clean_harm=harm)
    else:
        losses=[];rates=[]
        for d in (clean,stress):
            improvement=(F.cross_entropy(d['p0'].float(),y,reduction='none')-F.cross_entropy(d['p1'].float(),y,reduction='none')).detach()
            utility=(improvement>.02).float()
            losses.append(F.binary_cross_entropy_with_logits(d['utility_logit'].float().squeeze(1),utility))
            rates.append(utility.mean())
        loss=sum(losses)/2
        parts=dict(gate_bce=loss,utility_positive_clean=rates[0],utility_positive_corrupt=rates[1])
    assert torch.isfinite(loss),'Nonfinite loss'
    return loss,{k:float(v.detach()) for k,v in parts.items()}
