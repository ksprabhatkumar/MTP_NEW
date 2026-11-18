"""T10 matched temporal factorial. Runtime validation is mandatory before training."""
import torch
from experiments.t08_predictive_20261114.model import PredictiveRepair
from models.selective_mamba_temporal import SelectiveMambaTemporal

ARMS = ('none_clean', 'none_robust', 'physical_robust', 'local_repair', 'graph_repair')
TEMPORALS = ('old', 'ssm')

class TemporalFactorial(PredictiveRepair):
    def __init__(self, arm, temporal):
        if arm not in ARMS or temporal not in TEMPORALS:
            raise ValueError((arm, temporal))
        super().__init__('none_robust' if arm == 'physical_robust' else arm)
        self.spatial_arm = arm
        self.temporal_kind = temporal
        # Construct the common modules first: temporal choice must not change
        # initialization of shared spatial, auxiliary, or classifier parameters.
        if temporal == 'ssm':
            self.mamba = SelectiveMambaTemporal()

    def local_encode(self, x, a):
        if self.spatial_arm != 'physical_robust':
            return super().local_encode(x, a)
        b, t, n, _ = x.shape
        x = x * a[:, None, :, None]
        adjacency = self.physical[None] * a[:, None, :]
        adjacency = self.row_normalize(adjacency)
        adjacency = adjacency[:, None].expand(-1, t, -1, -1).reshape(b*t, n, n)
        z = self.encoder(x.reshape(b*t, n, -1), adjacency).reshape(b, t, n, -1)
        return z * a[:, None, :, None]


def optimizer_groups(model):
    return [dict(params=[p for p in model.parameters() if not getattr(p, '_no_weight_decay', False)], weight_decay=5e-4),
            dict(params=[p for p in model.parameters() if getattr(p, '_no_weight_decay', False)], weight_decay=0.)]
