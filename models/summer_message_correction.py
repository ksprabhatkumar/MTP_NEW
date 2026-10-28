"""O1 operator ablation: preserve self encoding, then add latent messages.

The temporal backbone is unchanged. `local_control` uses the same correction
network without neighbors. The final correction projection starts at zero.
"""
import torch
from torch import nn
from models.summer_dynamic import SummerDynamicWHAR


ARMS = ('none', 'local_control', 'message_correction')


class SummerMessageCorrection(SummerDynamicWHAR):
    def __init__(self, arm, hidden_dim=16):
        if arm not in ARMS:
            raise ValueError(arm)
        super().__init__(in_features=5, hidden_dim=hidden_dim,
                         graph_variant='residual', use_reconstruction=False)
        self.arm = arm
        # Allocate identically across arms to preserve shared initialization.
        self.message_value = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.correction = nn.Sequential(nn.Linear(2 * hidden_dim, hidden_dim),
                                        nn.SiLU(), nn.Linear(hidden_dim, hidden_dim, bias=False))
        nn.init.zeros_(self.correction[-1].weight)

    def forward(self, x, availability=None, return_diagnostics=False):
        b, t, n, _ = x.shape
        if availability is None:
            availability = torch.ones(b, n, device=x.device, dtype=torch.bool)
        if availability.shape != (b, n) or not availability.any(-1).all():
            raise ValueError('At least one available sensor is required per window')
        x = x * availability[:, None, :, None]
        mask = availability[:, None, :].expand(b, t, n).reshape(b*t, n, 1)
        zero = x.new_zeros(b*t, n, n)
        local = self.encoder(x.reshape(b*t, n, -1), zero) * mask
        adjacency = zero
        message = torch.zeros_like(local)
        delta = torch.zeros_like(local)
        if self.arm != 'none':
            values = self.message_value(local) * mask
            if self.arm == 'message_correction':
                adjacency, _ = self.graph(x, availability)
                message = torch.bmm(adjacency.to(values.dtype), values)
            else:
                message = values
            delta = self.correction(torch.cat((local, message), -1)) * mask
        latent = (local + delta) * mask
        temporal = self.mamba(latent.reshape(b, t, -1))
        temporal = temporal.reshape(b, t, n, self.hidden_dim) * availability[:, None, :, None]
        logits = self.classifier(self.norm_final(temporal.reshape(b, t, -1).mean(1)))
        if return_diagnostics:
            return logits, None, {'adjacency': adjacency.reshape(b, t, n, n),
                'local': local, 'message': message, 'correction': delta}
        return logits, None
