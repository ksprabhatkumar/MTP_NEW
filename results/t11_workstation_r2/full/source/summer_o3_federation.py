"""Subject-client federated primitives for the summer-based O3 simulator.

The normalization bridge and model arms are provisional until O2 is analyzed.
No GPU training is launched by importing this module.
"""

from collections import OrderedDict

import torch
from torch import nn

from models.summer_dynamic import SummerDynamicWHAR
from o2_features import WIDTHS


class FederatedSummerModel(nn.Module):
    """Summer graph model with a matched projection/normalization bridge."""

    def __init__(self, kind, graph, width=16):
        super().__init__()
        if kind not in ('raw', 'magnitude', 'mag_dot'):
            raise ValueError(kind)
        if graph not in ('physical', 'residual', 'none'):
            raise ValueError(graph)
        self.kind, self.graph = kind, graph
        self.input_projection = nn.Linear(WIDTHS[kind], width)
        self.input_bn = nn.BatchNorm1d(width)
        self.backbone = SummerDynamicWHAR(in_features=width,
            graph_variant=graph, use_reconstruction=False)

    def forward(self, x, availability, return_diagnostics=False):
        b, t, n, _ = x.shape
        projected = self.input_projection(x).reshape(b * t * n, -1)
        projected = self.input_bn(projected).reshape(b, t, n, -1)
        projected = projected * availability[:, None, :, None]
        return self.backbone(projected, availability,
                             return_diagnostics=return_diagnostics)


def local_normalization_keys(model):
    state = model.state_dict()
    keys = set()
    for name, module in model.named_modules():
        if isinstance(module, nn.modules.batchnorm._BatchNorm):
            prefix = name + '.' if name else ''
            keys.update(key for key in state if key.startswith(prefix))
    if not keys:
        raise ValueError('Client-local normalization requested without BatchNorm')
    return frozenset(keys)


def weighted_state_average(states, counts, exclude=(), reference=None):
    """FedAvg of parameters and buffers, retaining excluded reference keys."""
    if not states or len(states) != len(counts) or any(count <= 0 for count in counts):
        raise ValueError('One positive sample count per client is required')
    keys = tuple(states[0])
    if any(tuple(state) != keys for state in states[1:]):
        raise ValueError('Client state dictionaries have different keys')
    excluded = set(exclude)
    if excluded and reference is None:
        raise ValueError('A server reference state is required for local keys')
    total = float(sum(counts))
    result = OrderedDict()
    for key in keys:
        if key in excluded:
            result[key] = reference[key].clone()
            continue
        first = states[0][key]
        if first.is_floating_point():
            average = torch.zeros_like(first, dtype=torch.float64)
            for state, count in zip(states, counts):
                average += state[key].to(torch.float64) * (count / total)
            result[key] = average.to(first.dtype)
        else:
            # BatchNorm counters are the only integer buffers in this model.
            result[key] = torch.tensor(round(sum(
                int(state[key]) * count for state, count in zip(states, counts))
                / total), dtype=first.dtype, device=first.device)
    return result


def apply_local_state(shared, local, local_keys):
    if shared.keys() != local.keys():
        raise ValueError('Shared/local state dictionaries differ')
    return OrderedDict((key, local[key].clone() if key in local_keys
                        else value.clone()) for key, value in shared.items())


def state_bytes(state, exclude=()):
    excluded = set(exclude)
    return sum(value.numel() * value.element_size()
               for key, value in state.items() if key not in excluded)


def round_communication_bytes(shared_state, participating_clients,
                              local_keys=()):
    # One broadcast and one returned model update per participating client.
    return 2 * participating_clients * state_bytes(shared_state, local_keys)
