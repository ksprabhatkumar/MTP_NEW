"""Receiver-blind cross-sensor prediction, with matched local/no-graph controls."""
import torch
from torch import nn
from torch.nn import functional as F
from models.summer_dynamic import SummerDynamicWHAR
ARMS=('none_clean','none_robust','local_repair','graph_repair')

class PredictiveRepair(SummerDynamicWHAR):
    def __init__(self,arm,hidden_dim=16):
        if arm not in ARMS: raise ValueError(arm)
        super().__init__(in_features=5,hidden_dim=hidden_dim,graph_variant='residual',use_reconstruction=False)
        self.arm=arm
        # Pointwise LayerNorm here prevents receiver leakage through batch statistics.
        # The original classification encoder and its BatchNorm are unchanged.
        self.sender_encoder=nn.Sequential(nn.Linear(5,hidden_dim),nn.LayerNorm(hidden_dim),nn.SiLU())
        self.pair_embedding=nn.Embedding(81,4)
        self.relation_value=nn.Sequential(nn.Linear(hidden_dim+4,hidden_dim),nn.SiLU(),nn.Linear(hidden_dim,hidden_dim,bias=False))
        self.edge_score=nn.Sequential(nn.Linear(hidden_dim+4,hidden_dim),nn.SiLU(),nn.Linear(hidden_dim,1))
        self.quality=nn.Sequential(nn.Linear(2*hidden_dim,hidden_dim),nn.SiLU(),nn.Linear(hidden_dim,1))
        self.prediction_output=nn.Linear(hidden_dim,hidden_dim,bias=False)
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
        z=self.encoder(x.reshape(b*t,n,-1),x.new_zeros(b*t,n,n)).reshape(b,t,n,-1)
        return z*a[:,None,:,None]

    def forward(self,x,availability=None,return_diagnostics=False):
        b,t,n,_=x.shape
        a=availability if availability is not None else torch.ones(b,n,device=x.device,dtype=torch.bool)
        if a.shape!=(b,n) or not a.any(-1).all(): raise ValueError('At least one available sensor required')
        local=self.local_encode(x,a)
        z=self.sender_encoder(x*a[:,None,:,None])*a[:,None,:,None]
        descriptor=torch.cat((z.mean(1),z.float().std(1,unbiased=False).to(z.dtype)),-1)
        quality_logits=self.quality(descriptor).squeeze(-1)
        quality=torch.sigmoid(quality_logits)*a
        graph=self.arm=='graph_repair'
        # Shuffle the entire donor branch (features, quality, masks), never just values.
        donor=z;donor_quality=quality;donor_a=a
        if self.message_intervention=='window_shuffle':
            donor=z.roll(1,0);donor_quality=quality.roll(1,0);donor_a=a.roll(1,0)
        if self.message_intervention=='zero_message': donor=torch.zeros_like(donor)
        h=donor.mean(1)
        sender=h[:,None,:,:].expand(-1,n,-1,-1) if graph else h[:,:,None,:].expand(-1,-1,n,-1)
        emb=self.pair_embedding(torch.arange(n*n,device=x.device)).reshape(n,n,4)
        scores=self.edge_score(torch.cat((sender,emb[None].expand(b,-1,-1,-1)),-1)).squeeze(-1).float()
        mix=torch.softmax(self.branch_logits[:2],0)
        prior=self.row_normalize((mix[0]*self.physical+mix[1]*F.softplus(self.global_logits))*(1-torch.eye(n,device=x.device)))
        q=donor_quality[:,None,:] if graph else donor_quality[:,:,None]
        scores=scores+torch.log(prior.clamp_min(1e-8))[None]+torch.log(q.float().clamp_min(1e-8))
        valid=(~torch.eye(n,device=x.device,dtype=torch.bool))[None] & (donor_a[:,None,:] if graph else donor_a[:,:,None])
        scores=scores.masked_fill(~valid,-torch.inf)
        probs=torch.softmax(torch.cat((scores,self.null_logit[None,:,None].expand(b,-1,-1)),-1),-1)
        weights=probs[...,:n];null_mass=probs[...,-1]
        hj=donor[:,:,None,:,:].expand(-1,-1,n,-1,-1) if graph else donor[:,:,:,None,:].expand(-1,-1,-1,n,-1)
        e=emb[None,None].expand(b,t,-1,-1,-1)
        values=self.relation_value(torch.cat((hj,e),-1))-self.relation_value(torch.cat((torch.zeros_like(hj),e),-1))
        message=(weights[:,None,:,:,None].to(values.dtype)*values).sum(3)
        prediction=self.prediction_output(message)
        gate=torch.sigmoid(self.repair_gate(torch.cat((local.mean(1),prediction.mean(1)),-1)))
        correction=gate[:,None]*self.repair_output(prediction)*a[:,None,:,None]
        if self.arm in ('none_clean','none_robust') or self.message_intervention=='zero_correction':correction=torch.zeros_like(correction)
        latent=local+correction
        temporal=self.mamba(latent.reshape(b,t,-1)).reshape(b,t,n,-1)*a[:,None,:,None]
        logits=self.classifier(self.norm_final(temporal.reshape(b,t,-1).mean(1)))
        if return_diagnostics:
            return logits,None,dict(local=local,latent=latent,prediction=prediction,message=message,correction=correction,quality_logits=quality_logits,quality=quality,adjacency=weights,null_mass=null_mass,gate=gate)
        return logits,None
