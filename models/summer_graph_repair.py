"""Receiver-conditioned repair with sender reliability and a null neighbor."""
import torch
from torch import nn
from torch.nn import functional as F
from models.summer_dynamic import SummerDynamicWHAR

ARMS=('none_clean','none_robust','local_repair','graph_repair')

class SummerGraphRepair(SummerDynamicWHAR):
    def __init__(self,arm,hidden_dim=16):
        if arm not in ARMS: raise ValueError(arm)
        super().__init__(in_features=5,hidden_dim=hidden_dim,graph_variant='residual',use_reconstruction=False)
        self.arm=arm
        self.pair_embedding=nn.Embedding(81,4)
        self.relation_value=nn.Sequential(nn.Linear(2*hidden_dim+4,hidden_dim),nn.SiLU(),nn.Linear(hidden_dim,hidden_dim,bias=False))
        self.edge_score=nn.Sequential(nn.Linear(2*hidden_dim+4,hidden_dim),nn.SiLU(),nn.Linear(hidden_dim,1))
        self.quality=nn.Sequential(nn.Linear(2*hidden_dim,hidden_dim),nn.SiLU(),nn.Linear(hidden_dim,1))
        self.repair_gate=nn.Linear(2*hidden_dim,1)
        self.repair_output=nn.Linear(hidden_dim,hidden_dim,bias=False)
        self.null_logit=nn.Parameter(torch.full((9,),-2.))
        nn.init.zeros_(self.repair_output.weight)
        nn.init.zeros_(self.edge_score[-1].weight);nn.init.zeros_(self.edge_score[-1].bias)
        nn.init.zeros_(self.quality[-1].weight);nn.init.constant_(self.quality[-1].bias,2.)
        nn.init.zeros_(self.repair_gate.weight);nn.init.constant_(self.repair_gate.bias,2.)
        self.message_intervention=None

    def local_encode(self,x,a):
        b,t,n,_=x.shape
        x=x*a[:,None,:,None]
        local=self.encoder(x.reshape(b*t,n,-1),x.new_zeros(b*t,n,n)).reshape(b,t,n,-1)
        return local*a[:,None,:,None]

    def forward(self,x,availability=None,return_diagnostics=False):
        b,t,n,_=x.shape
        a=availability if availability is not None else torch.ones(b,n,device=x.device,dtype=torch.bool)
        if a.shape!=(b,n) or not a.any(-1).all():raise ValueError('At least one available sensor required')
        local=self.local_encode(x,a)
        descriptor=torch.cat((local.mean(1),local.float().std(1,unbiased=False).to(local.dtype)),-1)
        quality_logits=self.quality(descriptor).squeeze(-1)
        quality=torch.sigmoid(quality_logits)*a
        message=torch.zeros_like(local);delta=torch.zeros_like(local)
        weights=local.new_zeros(b,n,n);null_mass=local.new_ones(b,n);gate=local.new_zeros(b,n,1)
        if self.arm in ('local_repair','graph_repair'):
            h=local.mean(1)
            receiver=h[:,:,None,:].expand(-1,-1,n,-1)
            sender=h[:,None,:,:].expand(-1,n,-1,-1) if self.arm=='graph_repair' else receiver
            embedding=self.pair_embedding(torch.arange(n*n,device=x.device)).reshape(n,n,4)
            emb=embedding[None].expand(b,-1,-1,-1)
            scores=self.edge_score(torch.cat((receiver,sender,emb),-1)).squeeze(-1).float()
            mix=torch.softmax(self.branch_logits[:2],0)
            prior=mix[0]*self.physical+mix[1]*F.softplus(self.global_logits)
            prior=prior*(1-torch.eye(n,device=x.device))
            prior=self.row_normalize(prior)
            sender_quality=quality[:,None,:] if self.arm=='graph_repair' else quality[:,:,None]
            scores=scores+torch.log(prior.clamp_min(1e-8))[None]+torch.log(sender_quality.float().clamp_min(1e-8))
            valid=(~torch.eye(n,device=x.device,dtype=torch.bool))[None]
            valid=valid & (a[:,None,:] if self.arm=='graph_repair' else a[:,:,None])
            scores=scores.masked_fill(~valid,-torch.inf)
            probabilities=torch.softmax(torch.cat((scores,self.null_logit[None,:,None].expand(b,-1,-1)),-1),-1)
            weights=probabilities[...,:n];null_mass=probabilities[...,-1]
            hi=local[:,:,:,None,:].expand(-1,-1,-1,n,-1)
            hj=local[:,:,None,:,:].expand(-1,-1,n,-1,-1) if self.arm=='graph_repair' else hi
            if self.message_intervention=='zero_message':hj=torch.zeros_like(hj)
            elif self.message_intervention=='window_shuffle':hj=hj.roll(1,0)
            e=embedding[None,None].expand(b,t,-1,-1,-1)
            values=self.relation_value(torch.cat((hi,hj,e),-1))-self.relation_value(torch.cat((hi,torch.zeros_like(hj),e),-1))
            message=(weights[:,None,:,:,None].to(values.dtype)*values).sum(3)
            gate=torch.sigmoid(self.repair_gate(torch.cat((h,message.mean(1)),-1)))
            delta=gate[:,None]*self.repair_output(message)
            if self.message_intervention=='zero_correction':delta=torch.zeros_like(delta)
            delta=delta*a[:,None,:,None]
        latent=local+delta
        temporal=self.mamba(latent.reshape(b,t,-1)).reshape(b,t,n,-1)*a[:,None,:,None]
        logits=self.classifier(self.norm_final(temporal.reshape(b,t,-1).mean(1)))
        if return_diagnostics:
            return logits,None,dict(local=local,latent=latent,message=message,correction=delta,quality_logits=quality_logits,
                quality=quality,adjacency=weights,null_mass=null_mass,gate=gate)
        return logits,None
