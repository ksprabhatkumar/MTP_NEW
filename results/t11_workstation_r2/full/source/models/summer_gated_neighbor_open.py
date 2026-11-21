"""Gate-initialization ablation: same operator, initial acceptance sigmoid(2)."""
from torch import nn
from models.summer_gated_neighbor import ARMS, SummerGatedNeighbor

class SummerOpenGatedNeighbor(SummerGatedNeighbor):
    def __init__(self, arm, hidden_dim=16):
        super().__init__(arm, hidden_dim)
        nn.init.constant_(self.receiver_gate.bias, 2.)
