"""Adapter that runs the official DynamicWHAR model on Objective 1 tensors.

The upstream source is vendored in ``baselines/DynamicWHAR`` without modifying it.
This wrapper only adapts O1's [B, T, N, F] convention and builds its directed,
no-self-edge receiver/sender matrices.
"""

from pathlib import Path
import sys

import torch
import torch.nn as nn

UPSTREAM_ROOT = Path(__file__).with_name("DynamicWHAR")
if str(UPSTREAM_ROOT) not in sys.path:
    sys.path.insert(0, str(UPSTREAM_ROOT))

from modules import DynamicWHAR  # noqa: E402


class BatchSafeDynamicWHAR(DynamicWHAR):
    """Preserve upstream algebra while fixing its batch-one squeeze bug."""
    def edge2node(self, x, rel_rec, rel_send, rel_type):
        mask = rel_type.squeeze(-1)
        x = x + x * mask.unsqueeze(-1)
        incoming = torch.matmul(rel_rec.t() + rel_send.t(), x)
        return incoming / incoming.size(1)


def directed_relation_matrices(nodes, device=None, dtype=torch.float32):
    pairs = [(receiver, sender) for receiver in range(nodes) for sender in range(nodes) if receiver != sender]
    receivers = torch.zeros(len(pairs), nodes, device=device, dtype=dtype)
    senders = torch.zeros_like(receivers)
    for edge, (receiver, sender) in enumerate(pairs):
        receivers[edge, receiver] = 1
        senders[edge, sender] = 1
    return receivers, senders


class Objective1DynamicWHAR(nn.Module):
    """Official architecture with declared O1-compatible input dimensions."""
    def __init__(self, nodes=9, features=6, window=50, classes=33):
        super().__init__()
        self.model = BatchSafeDynamicWHAR(node_num=nodes, node_dim=features, window_size=window, class_num=classes)
        rel_rec, rel_send = directed_relation_matrices(nodes)
        self.register_buffer("rel_rec", rel_rec)
        self.register_buffer("rel_send", rel_send)

    def forward(self, x, availability=None, return_diagnostics=False):
        if availability is not None:
            if availability.ndim == 2:
                x = x * availability[:, None, :, None].to(x.dtype)
            else:
                x = x * availability[..., None].to(x.dtype)
        # Upstream expects [B,N,T,F]; this project supplies [B,T,N,F].
        logits = self.model(x.permute(0, 2, 1, 3).contiguous(), self.rel_rec, self.rel_send)
        if return_diagnostics:
            return logits, None, {"comparator": "official_dynamicwhar", "edge_count": self.rel_rec.size(0)}
        return logits, None
