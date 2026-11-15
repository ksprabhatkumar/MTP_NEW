"""Neighbor-only residual values with receiver-conditioned rejection.

Local features control the gate but cannot enter the correction value path.
All arms allocate identical modules and share initialization.
"""
import torch
from torch import nn
from models.summer_message_correction import SummerMessageCorrection

ARMS = ('none', 'local_control', 'gated_neighbor')

class SummerGatedNeighbor(SummerMessageCorrection):
    def __init__(self, arm, hidden_dim=16):
        if arm not in ARMS: raise ValueError(arm)
        super().__init__('none', hidden_dim)
        self.arm = arm
        self.neighbor_output = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.receiver_gate = nn.Linear(3*hidden_dim, 1)
        nn.init.zeros_(self.neighbor_output.weight)
        nn.init.zeros_(self.receiver_gate.weight)
        nn.init.constant_(self.receiver_gate.bias, -2.)
        self.message_intervention = None

    def forward(self, x, availability=None, return_diagnostics=False):
        b,t,n,_ = x.shape
        if availability is None:
            availability = torch.ones(b,n,device=x.device,dtype=torch.bool)
        if availability.shape != (b,n) or not availability.any(-1).all():
            raise ValueError('At least one available sensor is required')
        x = x*availability[:,None,:,None]
        mask = availability[:,None,:].expand(b,t,n).reshape(b*t,n,1)
        adjacency = x.new_zeros(b*t,n,n)
        local = self.encoder(x.reshape(b*t,n,-1),adjacency)*mask
        message,delta = torch.zeros_like(local),torch.zeros_like(local)
        gate = local.new_zeros(b*t,n,1)
        if self.arm != 'none':
            values = self.message_value(local)*mask
            if self.arm == 'gated_neighbor':
                adjacency,_ = self.graph(x,availability)
                message = torch.bmm(adjacency.to(values.dtype),values)
            else:
                message = values
            if self.message_intervention == 'zero_message': message = torch.zeros_like(message)
            elif self.message_intervention == 'window_shuffle':
                message = message.reshape(b,t,n,-1).roll(1,0).reshape(b*t,n,-1)
            gate = torch.sigmoid(self.receiver_gate(torch.cat((local,message,(local-message).abs()),-1)))
            delta = gate*self.neighbor_output(torch.nn.functional.silu(message))*mask
            if self.message_intervention == 'zero_correction': delta = torch.zeros_like(delta)
        latent = (local+delta)*mask
        temporal = self.mamba(latent.reshape(b,t,-1)).reshape(b,t,n,self.hidden_dim)
        temporal = temporal*availability[:,None,:,None]
        logits = self.classifier(self.norm_final(temporal.reshape(b,t,-1).mean(1)))
        if return_diagnostics:
            return logits,None,dict(adjacency=adjacency.reshape(b,t,n,n),local=local,
                                   message=message,correction=delta,gate=gate)
        return logits,None
