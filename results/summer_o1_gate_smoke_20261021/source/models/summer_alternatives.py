"""Isolated O1 graph follow-ups on the frozen summer backbone.

These are project adaptations, not reproductions of DecomposeWHAR or 2s-AGCN.
The original graph implementation remains unchanged for historical controls.
"""

import math

import torch
from torch import nn
from torch.nn import functional as F

from models.summer_dynamic import SummerDynamicWHAR


MODES = {"masked_balanced", "masked_floor", "pairwise_balanced", "pairwise_floor",
         "gated_masked", "gated_pairwise"}


class SummerAlternativeGraph(SummerDynamicWHAR):
    def __init__(self, mode, in_features=5):
        if mode not in MODES:
            raise ValueError(mode)
        super().__init__(in_features=in_features, graph_variant="residual",
                         use_reconstruction=False)
        self.router_mode = mode
        if "pairwise" in mode:
            width = self.hidden_dim
            self.pairwise = nn.Sequential(nn.Linear(2 * width, width),
                                          nn.SiLU(), nn.Linear(width, 1))
        if mode.startswith("gated"):
            self.neighbor_gate = nn.Sequential(nn.Linear(self.hidden_dim, 1),
                                               nn.Sigmoid())

    def graph(self, x, available):
        batch, steps, nodes, _ = x.shape
        mask = available[:, None, :].expand(batch, steps, nodes).reshape(batch * steps, nodes)
        eligible = (mask[:, :, None] & mask[:, None, :])
        eligible = eligible & ~torch.eye(nodes, device=x.device, dtype=torch.bool)[None]
        encoded = self.affinity_projection(x)
        encoded = encoded.permute(0, 2, 3, 1).reshape(batch * nodes, self.hidden_dim, steps)
        encoded = F.silu(self.affinity_temporal(encoded)[..., :steps])
        encoded = (encoded.reshape(batch, nodes, self.hidden_dim, steps)
                   .permute(0, 3, 1, 2).reshape(batch * steps, nodes, self.hidden_dim))
        encoded = encoded * mask[..., None]
        if "pairwise" in self.router_mode:
            receiver = encoded[:, :, None, :].expand(-1, -1, nodes, -1)
            sender = encoded[:, None, :, :].expand(-1, nodes, -1, -1)
            scores = self.pairwise(torch.cat((receiver, sender), -1)).squeeze(-1).float()
        else:
            scores = torch.bmm(self.q(encoded).float(),
                               self.k(encoded).float().transpose(1, 2)) / math.sqrt(self.hidden_dim)
        valid_row = eligible.any(-1, keepdim=True)
        dynamic = torch.softmax(scores.masked_fill(~eligible, -1e9), -1)
        dynamic = torch.where(valid_row, dynamic, torch.zeros_like(dynamic))
        if self.intervention == "uniform":
            dynamic = eligible.float() / eligible.sum(-1, keepdim=True).clamp_min(1)
        elif self.intervention == "shuffled":
            dynamic = dynamic.roll(1, 0) * eligible
            dynamic = self.row_normalize(dynamic)
        physical = self.row_normalize(self.physical[None] * eligible)
        global_graph = self.row_normalize(F.softplus(self.global_logits)[None] * eligible)
        components = torch.stack((physical, global_graph, dynamic), 1)
        weights = torch.softmax(self.branch_logits, 0)
        if self.router_mode.endswith("floor"):
            floor = torch.tensor([0., 0., 1.], device=x.device)
            weights = .75 * weights + .25 * floor
        active = components.sum(-1) > 0
        row_weights = weights[None, :, None] * active
        row_weights = row_weights / row_weights.sum(1, keepdim=True).clamp_min(1e-12)
        mixed = components * row_weights[..., None]
        adjacency = mixed.sum(1)
        if self.router_mode.startswith("gated"):
            gate = self.neighbor_gate(encoded).squeeze(-1) * mask
            adjacency = adjacency * gate[..., None]
            mixed = mixed * gate[:, None, :, None]
        else:
            gate = None
        effective = mixed.sum(-1).mean((0, 2))
        return adjacency, {"adjacency": adjacency.reshape(batch, steps, nodes, nodes),
                           "dynamic": dynamic.reshape(batch, steps, nodes, nodes),
                           "branch_weights": weights,
                           "effective_branch_mass": effective,
                           "neighbor_gate_mean": gate.mean() if gate is not None else None,
                           "encoded": encoded}
