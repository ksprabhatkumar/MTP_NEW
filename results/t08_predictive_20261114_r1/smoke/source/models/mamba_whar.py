"""Models for Objective 1: placement-aware residual inter-sensor routing."""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualDynamicGraph(nn.Module):
    """Build a masked, row-normalized physical/global/dynamic sensor graph."""

    VALID_VARIANTS = {"none", "physical", "global", "dynamic", "residual"}

    def __init__(self, num_nodes, in_features, attention_dim=None, temperature=1.0):
        super().__init__()
        self.num_nodes = num_nodes
        attention_dim = attention_dim or in_features
        self.q_proj = nn.Linear(in_features, attention_dim, bias=False)
        self.k_proj = nn.Linear(in_features, attention_dim, bias=False)
        self.global_logits = nn.Parameter(torch.zeros(num_nodes, num_nodes))
        # Softmax gives non-negative weights summing to one and initially favours anatomy.
        self.branch_logits = nn.Parameter(torch.tensor([2.0, 0.0, 0.0]))
        self.temperature = float(temperature)

    def forward(self, x, physical_prior, availability=None, variant="residual"):
        if variant not in self.VALID_VARIANTS:
            raise ValueError(f"Unknown graph variant: {variant}")
        batch, nodes, _ = x.shape
        if nodes != self.num_nodes:
            raise ValueError(f"Expected {self.num_nodes} sensor nodes, received {nodes}")

        if availability is None:
            availability = torch.ones(batch, nodes, device=x.device, dtype=torch.bool)
        else:
            availability = availability.to(device=x.device, dtype=torch.bool)
            if availability.shape != (batch, nodes):
                raise ValueError("availability must have shape [batch, num_nodes]")

        off_diagonal = (~torch.eye(nodes, dtype=torch.bool, device=x.device)).unsqueeze(0)
        sender_mask = availability.unsqueeze(1).expand(-1, nodes, -1) & off_diagonal

        physical = F.relu(physical_prior).to(dtype=x.dtype, device=x.device)
        physical = physical.unsqueeze(0).expand(batch, -1, -1) * off_diagonal
        global_branch = F.softplus(self.global_logits).unsqueeze(0).expand(batch, -1, -1)
        global_branch = global_branch * off_diagonal

        q = self.q_proj(torch.tanh(x))
        k = self.k_proj(torch.tanh(x))
        scores = torch.bmm(q, k.transpose(1, 2)) / (self.temperature * math.sqrt(q.size(-1)))
        scores = scores.masked_fill(~sender_mask, -torch.inf)
        # A row with no valid senders intentionally contributes no dynamic messages.
        valid_row = sender_mask.any(dim=-1, keepdim=True)
        safe_scores = torch.where(valid_row, scores, torch.zeros_like(scores))
        dynamic = torch.softmax(safe_scores, dim=-1) * sender_mask

        branch_weights = torch.softmax(self.branch_logits, dim=0)
        if variant == "none":
            unnormalized = torch.zeros_like(dynamic)
        elif variant == "physical":
            unnormalized = physical
        elif variant == "global":
            unnormalized = global_branch
        elif variant == "dynamic":
            unnormalized = dynamic
        else:
            unnormalized = (
                branch_weights[0] * physical
                + branch_weights[1] * global_branch
                + branch_weights[2] * dynamic
            )
        unnormalized = unnormalized * sender_mask
        row_sum = unnormalized.sum(dim=-1, keepdim=True)
        adjacency = torch.where(row_sum > 0, unnormalized / row_sum.clamp_min(1e-12), torch.zeros_like(unnormalized))
        diagnostics = {
            "adjacency": adjacency,
            "physical": physical,
            "global": global_branch,
            "dynamic": dynamic,
            "branch_weights": branch_weights,
            "masked_rows": (~valid_row.squeeze(-1)).sum(),
        }
        return adjacency, diagnostics


class KinematicGraphEncoder(nn.Module):
    """Weighted neighbour messages plus an explicit residual self-feature path."""

    def __init__(self, in_features, hidden_dim):
        super().__init__()
        self.self_proj = nn.Linear(in_features, hidden_dim)
        self.message_proj = nn.Linear(in_features, hidden_dim, bias=False)
        self.norm = nn.BatchNorm1d(hidden_dim)

    def forward(self, x, adjacency):
        messages = torch.bmm(adjacency, x)
        encoded = self.self_proj(x) + self.message_proj(messages)
        batch, nodes, width = encoded.shape
        encoded = self.norm(encoded.reshape(batch * nodes, width))
        return F.relu(encoded).reshape(batch, nodes, width)


class KinematicGraphDecoder(nn.Module):
    def __init__(self, hidden_dim, out_features):
        super().__init__()
        self.self_proj = nn.Linear(hidden_dim, hidden_dim)
        self.message_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.output = nn.Linear(hidden_dim, out_features)

    def forward(self, latent_x, adjacency):
        decoded = F.relu(self.self_proj(latent_x) + self.message_proj(torch.bmm(adjacency, latent_x)))
        return self.output(decoded)


class PurePyTorchMambaBlock(nn.Module):
    def __init__(self, d_model, expand=2, d_conv=4, dropout=0.3):
        super().__init__()
        self.d_inner = d_model * expand
        self.norm = nn.LayerNorm(d_model)
        self.in_proj = nn.Linear(d_model, self.d_inner * 2)
        self.conv1d = nn.Conv1d(self.d_inner, self.d_inner, kernel_size=d_conv, groups=self.d_inner, padding=d_conv - 1)
        self.ssm_proj = nn.Linear(self.d_inner, self.d_inner)
        self.out_proj = nn.Linear(self.d_inner, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        residual = x
        projected, gate = self.in_proj(self.norm(x)).chunk(2, dim=-1)
        length = projected.size(1)
        convolved = self.conv1d(projected.transpose(1, 2))[:, :, :length].transpose(1, 2)
        output = self.out_proj(self.ssm_proj(F.silu(convolved)) * F.silu(gate))
        return residual + self.dropout(output)


class GAE_Mamba_WHAR(nn.Module):
    """Objective 1 classifier. ``return_diagnostics`` keeps legacy callers compatible."""

    def __init__(self, num_nodes=9, in_features=6, hidden_dim=64, num_classes=33, dropout=0.3, graph_variant="residual", use_reconstruction=False):
        super().__init__()
        if num_nodes != 9:
            raise ValueError("Objective 1 currently supports REALDISP's nine sensor nodes only")
        self.num_nodes = num_nodes
        self.hidden_dim = hidden_dim
        self.graph_variant = graph_variant
        self.use_reconstruction = use_reconstruction
        edges = [(0, 1), (1, 0), (0, 2), (2, 0), (0, 5), (5, 0), (0, 6), (6, 0), (1, 3), (3, 1), (2, 4), (4, 2), (5, 7), (7, 5), (6, 8), (8, 6)]
        prior = torch.zeros(num_nodes, num_nodes)
        for receiver, sender in edges:
            prior[receiver, sender] = 1.0
        self.register_buffer("A_phys", prior)
        self.dynamic_graph = ResidualDynamicGraph(num_nodes, in_features, hidden_dim)
        self.encoder = KinematicGraphEncoder(in_features, hidden_dim)
        self.decoder = KinematicGraphDecoder(hidden_dim, in_features) if use_reconstruction else None
        width = num_nodes * hidden_dim
        self.mamba = PurePyTorchMambaBlock(width, dropout=dropout)
        self.norm_final = nn.LayerNorm(width)
        self.classifier = nn.Linear(width, num_classes)

    def forward(self, x, availability=None, return_diagnostics=False):
        batch, steps, nodes, features = x.shape
        if nodes != self.num_nodes:
            raise ValueError(f"Expected {self.num_nodes} nodes, received {nodes}")
        flat = x.reshape(batch * steps, nodes, features)
        if availability is None:
            flat_availability = None
        else:
            if availability.ndim == 2:
                flat_availability = availability[:, None, :].expand(-1, steps, -1).reshape(batch * steps, nodes)
            elif availability.ndim == 3:
                flat_availability = availability.reshape(batch * steps, nodes)
            else:
                raise ValueError("availability must have shape [B,N] or [B,T,N]")
        adjacency, diagnostics = self.dynamic_graph(flat, self.A_phys, flat_availability, self.graph_variant)
        latent = self.encoder(flat, adjacency)
        reconstruction = self.decoder(latent, adjacency).reshape(batch, steps, nodes, features) if self.decoder else None
        temporal = self.mamba(latent.reshape(batch, steps, -1))
        logits = self.classifier(self.norm_final(temporal.mean(dim=1)))
        if return_diagnostics:
            diagnostics = {key: (value.reshape(batch, steps, *value.shape[1:]) if key in {"adjacency", "physical", "global", "dynamic"} else value) for key, value in diagnostics.items()}
            return logits, reconstruction, diagnostics
        return logits, reconstruction
