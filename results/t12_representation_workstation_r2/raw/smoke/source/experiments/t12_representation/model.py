"""Input-width ablation retaining T11's shared spatial and temporal weights."""
import torch
from torch import nn
from experiments.t10_temporal.model import TemporalFactorial as Base,optimizer_groups
from .common import representation

class TemporalFactorial(Base):
    def __init__(self,arm,temporal):
        super().__init__(arm,temporal)
        self.representation=representation()
        width={'raw':6,'magnitude':2}[self.representation]
        # Preserve all common weights and the caller's RNG. Only the three
        # active input projections change dimension. Unused inherited decoder
        # and affinity modules remain allocated, exactly as in the T11 control.
        with torch.random.fork_rng(devices=[]):
            torch.default_generator.manual_seed(torch.initial_seed()+40000)
            self.encoder.gin_agg.nn[0]=nn.Linear(width,16)
            self.encoder.gin_att.nn[0]=nn.Linear(width,16)
            self.sender_encoder[0]=nn.Linear(width,16)
