"""Summer GIN-attention pipeline with an isolated, weighted topology change.

Original source: baselines/summer_original/PROVENANCE.json. DenseGIN preserves
GINConv's (1+eps)*self + neighbor sum before the original MLP. The temporal
algebra/head are unchanged. Learned per-IMU temporal descriptors are used only
to estimate adjacency; the summer message/attention MLPs retain their inputs.
The principal residual adjacency implements objectives_methodology.tex exactly;
branch-balanced mixing is explicitly a separate ablation.
"""
import math
import torch
from torch import nn
from torch.nn import functional as F
from models.mamba_whar import PurePyTorchMambaBlock


class DenseGIN(nn.Module):
    def __init__(self, network):
        super().__init__()
        self.nn = network
        self.eps = nn.Parameter(torch.zeros(1))

    def forward(self, x, adjacency):
        combined = (1 + self.eps) * x + torch.bmm(adjacency.to(x.dtype), x)
        shape = combined.shape
        return self.nn(combined.reshape(-1, shape[-1])).reshape(shape[0], shape[1], -1)


class SummerEncoder(nn.Module):
    def __init__(self, inputs, hidden):
        super().__init__()
        self.gin_agg = DenseGIN(nn.Sequential(nn.Linear(inputs, hidden), nn.BatchNorm1d(hidden), nn.ReLU()))
        self.gin_att = DenseGIN(nn.Sequential(nn.Linear(inputs, hidden), nn.Tanh()))

    def forward(self, x, adjacency):
        return self.gin_agg(x, adjacency) * self.gin_att(x, adjacency)


class SummerDecoder(nn.Module):
    def __init__(self, hidden, outputs):
        super().__init__()
        self.gin_dec = DenseGIN(nn.Sequential(nn.Linear(hidden, hidden), nn.ReLU(), nn.Linear(hidden, outputs)))

    def forward(self, x, adjacency):
        return self.gin_dec(x, adjacency)


class SummerDynamicWHAR(nn.Module):
    VARIANTS = {'static_sum', 'physical', 'global', 'dynamic', 'residual', 'none', 'residual_balanced'}

    def __init__(self, in_features=6, hidden_dim=16, dropout=.5,
                 graph_variant='residual', use_reconstruction=False, num_nodes=9, num_classes=33):
        super().__init__()
        if graph_variant not in self.VARIANTS or num_nodes != 9:
            raise ValueError('Unsupported graph or node count')
        self.num_nodes, self.hidden_dim = num_nodes, hidden_dim
        self.graph_variant = graph_variant
        self.encoder = SummerEncoder(in_features, hidden_dim)
        # Allocate in every variant so shared weights initialize identically.
        self.decoder = SummerDecoder(hidden_dim, in_features)
        self.use_reconstruction = use_reconstruction
        width = num_nodes * hidden_dim
        self.mamba = PurePyTorchMambaBlock(width, dropout=dropout)
        self.norm_final = nn.LayerNorm(width)
        self.classifier = nn.Linear(width, num_classes)
        edges = [(0,1),(1,0),(0,2),(2,0),(0,5),(5,0),(0,6),(6,0),
                 (1,3),(3,1),(2,4),(4,2),(5,7),(7,5),(6,8),(8,6)]
        physical = torch.zeros(num_nodes, num_nodes)
        for receiver, sender in edges:
            physical[receiver, sender] = 1
        self.register_buffer('physical', physical)
        self.affinity_projection = nn.Sequential(nn.Linear(in_features, hidden_dim), nn.LayerNorm(hidden_dim), nn.SiLU())
        self.affinity_temporal = nn.Conv1d(hidden_dim, hidden_dim, 5, groups=hidden_dim, padding=4)
        self.q = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.k = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.global_logits = nn.Parameter(torch.zeros(num_nodes, num_nodes))
        self.branch_logits = nn.Parameter(torch.tensor([2., 0., 0.]))
        self.intervention = None

    @staticmethod
    def row_normalize(matrix):
        return matrix / matrix.sum(-1, keepdim=True).clamp_min(1e-12)

    def graph(self, x, available):
        b, t, n, _ = x.shape
        mask = available[:, None, :].expand(b, t, n).reshape(b*t, n)
        off = 1 - torch.eye(n, device=x.device)
        sender = mask[:, None, :].float() * off
        encoded = self.affinity_projection(x)
        encoded = encoded.permute(0,2,3,1).reshape(b*n, self.hidden_dim, t)
        encoded = F.silu(self.affinity_temporal(encoded)[..., :t])
        encoded = encoded.reshape(b,n,self.hidden_dim,t).permute(0,3,1,2).reshape(b*t,n,self.hidden_dim)
        encoded = encoded * mask[..., None]
        # C includes diagonal before masking, as in the authoritative equation.
        scores = torch.bmm(self.q(encoded).float(), self.k(encoded).float().transpose(1,2)) / math.sqrt(self.hidden_dim)
        dynamic = torch.softmax(scores, -1)
        if self.intervention == 'uniform':
            dynamic = torch.full_like(dynamic, 1 / n)
        elif self.intervention == 'shuffled':
            dynamic = dynamic.roll(1, 0)
        physical = self.physical.expand(b*t,-1,-1) * sender
        global_graph = F.softplus(self.global_logits).expand(b*t,-1,-1) * sender
        dynamic = dynamic * sender
        weights = torch.softmax(self.branch_logits, 0)
        components = torch.stack((physical, global_graph, dynamic), 1)
        if self.graph_variant == 'residual_balanced':
            components = self.row_normalize(components)
        if self.graph_variant in ('residual', 'residual_balanced'):
            mixed_components = components * weights[None,:,None,None]
            combined = mixed_components.sum(1)
        else:
            index = {'static_sum':0, 'physical':0, 'global':1, 'dynamic':2}.get(self.graph_variant)
            combined = torch.zeros_like(dynamic) if index is None else components[:,index]
            selector = torch.zeros(3,device=x.device)
            if index is not None: selector[index] = 1
            mixed_components = components * selector[None,:,None,None]
        adjacency = combined if self.graph_variant == 'static_sum' else self.row_normalize(combined)
        mass = mixed_components.sum(-1)
        effective = mass / mass.sum(1, keepdim=True).clamp_min(1e-12)
        return adjacency, {'adjacency':adjacency.reshape(b,t,n,n),
            'dynamic':dynamic.reshape(b,t,n,n), 'branch_weights':weights,
            'effective_branch_mass':effective.mean((0,2)), 'encoded':encoded}

    def forward(self, x, availability=None, return_diagnostics=False):
        b, t, n, f = x.shape
        if availability is None:
            availability = torch.ones(b,n,device=x.device,dtype=torch.bool)
        if availability.shape != (b,n) or not availability.any(-1).all():
            raise ValueError('Unavailable input: require at least one sensor per window (abstain otherwise).')
        x = x * availability[:,None,:,None]
        adjacency, diagnostics = self.graph(x, availability)
        mask = availability[:,None,:].expand(b,t,n).reshape(b*t,n,1)
        latent = self.encoder(x.reshape(b*t,n,f), adjacency) * mask
        reconstruction = self.decoder(latent, adjacency).reshape(b,t,n,f) if self.use_reconstruction else None
        temporal = self.mamba(latent.reshape(b,t,-1))
        temporal = temporal.reshape(b,t,n,self.hidden_dim) * availability[:,None,:,None]
        logits = self.classifier(self.norm_final(temporal.reshape(b,t,-1).mean(1)))
        if return_diagnostics:
            diagnostics['latent'] = latent.reshape(b,t,n,self.hidden_dim)
            return logits, reconstruction, diagnostics
        return logits, reconstruction
