"""Original correction unchanged, with inference-only intervention instrumentation."""
import torch
from models.summer_message_correction import ARMS,SummerMessageCorrection

class SummerReferenceCorrection(SummerMessageCorrection):
    def __init__(self,arm,hidden_dim=16):
        super().__init__(arm,hidden_dim)
        self.message_intervention=None

    def forward(self,x,availability=None,return_diagnostics=False):
        handles=[]
        if self.message_intervention in ('zero_message','window_shuffle'):
            def intervene(module,args):
                z=args[0].clone();h=z.shape[-1]//2
                if self.message_intervention=='zero_message':z[...,h:]=0
                else:z[...,h:]=z[...,h:].reshape(len(x),x.shape[1],9,h).roll(1,0).reshape(-1,9,h)
                return (z,)
            handles.append(self.correction.register_forward_pre_hook(intervene))
        elif self.message_intervention=='zero_correction':
            handles.append(self.correction.register_forward_hook(lambda module,args,out:torch.zeros_like(out)))
        try:
            result=super().forward(x,availability,return_diagnostics)
            if return_diagnostics:
                # Original model has an implicit multiplier of one, not a learned gate.
                result[2]['gate']=torch.ones_like(result[2]['local'][...,:1])
            return result
        finally:
            for handle in handles:handle.remove()
