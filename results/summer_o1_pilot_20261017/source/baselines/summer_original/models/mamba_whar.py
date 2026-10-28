import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GINConv

class KinematicGraphEncoder(nn.Module):
    """Encodes corrupted physical sensors into a Latent Kinematic Space"""
    def __init__(self, in_features, hidden_dim):
        super().__init__()
        agg_nn = nn.Sequential(
            nn.Linear(in_features, hidden_dim), 
            nn.BatchNorm1d(hidden_dim), 
            nn.ReLU()
        )
        self.gin_agg = GINConv(agg_nn, train_eps=True)
        
        att_nn = nn.Sequential(
            nn.Linear(in_features, hidden_dim), 
            nn.Tanh()
        )
        self.gin_att = GINConv(att_nn, train_eps=True)

    def forward(self, x, edge_index):
        node_features = self.gin_agg(x, edge_index)
        attention_weights = self.gin_att(x, edge_index)
        return node_features * attention_weights

class KinematicGraphDecoder(nn.Module):
    """Decodes the Latent Space back into raw sensor waves (Imputation)"""
    def __init__(self, hidden_dim, out_features):
        super().__init__()
        dec_nn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, out_features)
        )
        self.gin_dec = GINConv(dec_nn, train_eps=True)

    def forward(self, latent_x, edge_index):
        # Reconstructs the 9-feature sensor data
        return self.gin_dec(latent_x, edge_index)

class PurePyTorchMambaBlock(nn.Module):
    """Upgraded Pure PyTorch Mamba with Residuals, LayerNorm, and Dropout
       (Bypasses the need for Windows C++ compilers!)"""
    def __init__(self, d_model, expand=2, d_conv=4, dropout=0.3):
        super().__init__()
        self.d_inner = d_model * expand
        self.norm = nn.LayerNorm(d_model)
        
        self.in_proj = nn.Linear(d_model, self.d_inner * 2)
        self.conv1d = nn.Conv1d(
            in_channels=self.d_inner, out_channels=self.d_inner, 
            kernel_size=d_conv, groups=self.d_inner, padding=d_conv - 1
        )
        self.ssm_proj = nn.Linear(self.d_inner, self.d_inner)
        self.out_proj = nn.Linear(self.d_inner, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, L, D = x.shape
        residual = x
        x = self.norm(x)
        
        x_proj = self.in_proj(x)
        x_proj, res = x_proj.chunk(2, dim=-1)
        x_conv = x_proj.transpose(1, 2)
        x_conv = self.conv1d(x_conv)[:, :, :L].transpose(1, 2)
        x_conv = F.silu(x_conv)
        
        ssm_out = self.ssm_proj(x_conv) * F.silu(res)
        out = self.out_proj(ssm_out)
        out = self.dropout(out)
        
        return out + residual

class GAE_Mamba_WHAR(nn.Module):
    def __init__(self, num_nodes=5, in_features=9, hidden_dim=64, num_classes=19, dropout=0.3):
        super().__init__()
        self.num_nodes = num_nodes
        self.hidden_dim = hidden_dim
        
        # 🌟 DYNAMIC GRAPH TOPOLOGY
        if num_nodes == 5:
            edges = [[0, 1], [1, 0], [0, 2], [2, 0], [0, 3], [3, 0], [0, 4], [4, 0]]
        elif num_nodes == 9:
            edges = [
                [0, 1], [1, 0], [0, 2], [2, 0], [0, 5], [5, 0], [0, 6], [6, 0],
                [1, 3], [3, 1], [2, 4], [4, 2],
                [5, 7], [7, 5], [6, 8], [8, 6]
            ]
        else:
            raise ValueError("Unsupported number of nodes!")
            
        self.base_edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
        
        # 🌟 PHASE 2: GENERATIVE GRAPH AUTOENCODER (GAE)
        self.encoder = KinematicGraphEncoder(in_features, hidden_dim)
        self.decoder = KinematicGraphDecoder(hidden_dim, out_features=in_features)
        
        # Temporal Mamba
        mamba_dim = num_nodes * hidden_dim
        self.mamba = PurePyTorchMambaBlock(d_model=mamba_dim, dropout=dropout)
        
        self.norm_final = nn.LayerNorm(mamba_dim)
        self.classifier = nn.Linear(mamba_dim, num_classes)

    def get_batched_edge_index(self, batch_size, device):
        edge_index = self.base_edge_index.to(device)
        num_edges = edge_index.size(1)
        batched_edge_index = edge_index.repeat(1, batch_size)
        offset = torch.arange(0, batch_size, device=device) * self.num_nodes
        offset = offset.view(-1, 1).repeat(1, num_edges).view(1, -1)
        return batched_edge_index + offset

    def forward(self, x_corrupt):
        B, T, N, F_dim = x_corrupt.shape
        device = x_corrupt.device
        
        x_flat = x_corrupt.view(B * T * N, F_dim)
        edge_index = self.get_batched_edge_index(B * T, device)
        
        # 1. ENCODE: Compress corrupted graph to latent kinematics
        latent_graph = self.encoder(x_flat, edge_index)
        
        # 2. DECODE: Hallucinate missing sensors!
        x_recon_flat = self.decoder(latent_graph, edge_index)
        x_recon = x_recon_flat.view(B, T, N, F_dim)
        
        # 3. TEMPORAL PROCESSING: Mamba learns from the clean Latent space
        seq_x = latent_graph.view(B, T, N * self.hidden_dim)
        mamba_out = self.mamba(seq_x)
        
        # 4. CLASSIFICATION: Global Average Pooling
        pooled_state = mamba_out.mean(dim=1)
        final_state = self.norm_final(pooled_state)
        logits = self.classifier(final_state)
        
        # Return BOTH the guess AND the hallucinated sensor graph
        return logits, x_recon