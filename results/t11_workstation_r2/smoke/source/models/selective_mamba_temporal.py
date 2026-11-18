"""Transparent Mamba-1 selective recurrence; no optimized CUDA extension required.
Implements docs/experiments/mamba_temporal_spec_20261028.md. Validation pending.
"""
import math
import torch
from torch import nn
from torch.nn import functional as F

class SelectiveMambaTemporal(nn.Module):
    def __init__(self,d_model=144,d_state=16,expand=2,d_conv=4,dt_rank=9,dropout=.5,backend='reference'):
        super().__init__()
        if backend!='reference':raise ValueError('Only reference backend validated by this interface')
        self.d_model=d_model;self.d_state=d_state;self.d_inner=d_model*expand;self.dt_rank=dt_rank
        self.norm=nn.LayerNorm(d_model)
        self.in_proj=nn.Linear(d_model,2*self.d_inner,bias=False)
        self.conv1d=nn.Conv1d(self.d_inner,self.d_inner,d_conv,groups=self.d_inner,padding=d_conv-1)
        self.x_proj=nn.Linear(self.d_inner,dt_rank+2*d_state,bias=False)
        self.dt_proj=nn.Linear(dt_rank,self.d_inner)
        nn.init.uniform_(self.dt_proj.weight,-dt_rank**-.5,dt_rank**-.5)
        dt=torch.exp(torch.rand(self.d_inner)*(math.log(.1)-math.log(.001))+math.log(.001)).clamp_min(1e-4)
        with torch.no_grad():self.dt_proj.bias.copy_(dt+torch.log(-torch.expm1(-dt)))
        self.dt_proj.bias._no_reinit=True
        self.A_log=nn.Parameter(torch.arange(1,d_state+1,dtype=torch.float32).log().repeat(self.d_inner,1))
        self.D=nn.Parameter(torch.ones(self.d_inner))
        self.A_log._no_weight_decay=True;self.D._no_weight_decay=True
        self.out_proj=nn.Linear(self.d_inner,d_model,bias=False);self.dropout=nn.Dropout(dropout)

    def forward(self,x):
        if x.ndim!=3 or x.shape[-1]!=self.d_model:raise ValueError('Expected [B,T,d_model]')
        residual=x;u,z=self.in_proj(self.norm(x)).chunk(2,-1)
        u=F.silu(self.conv1d(u.transpose(1,2))[...,:x.shape[1]].transpose(1,2))
        dt,b,c=torch.split(self.x_proj(u),[self.dt_rank,self.d_state,self.d_state],dim=-1)
        # Keep delta projection and recurrence in FP32 even under outer autocast.
        with torch.autocast(device_type=x.device.type,enabled=False):
            delta=F.softplus(F.linear(dt.float(),self.dt_proj.weight.float(),self.dt_proj.bias.float()))
            a=-self.A_log.float().exp();state=torch.zeros(x.shape[0],self.d_inner,self.d_state,device=x.device,dtype=torch.float32)
            ys=[]
            for t in range(x.shape[1]):
                step=delta[:,t,:,None]
                state=torch.exp(step*a)*state+step*b[:,t,None,:].float()*u[:,t,:,None].float()
                ys.append((state*c[:,t,None,:].float()).sum(-1)+self.D.float()*u[:,t].float())
            y=torch.stack(ys,1)*F.silu(z.float())
        return residual+self.dropout(self.out_proj(y.to(u.dtype)))

    def optimizer_groups(self,weight_decay):
        decay=[];no_decay=[]
        for p in self.parameters():
            (no_decay if getattr(p,'_no_weight_decay',False) else decay).append(p)
        return [dict(params=decay,weight_decay=weight_decay),dict(params=no_decay,weight_decay=0.)]
