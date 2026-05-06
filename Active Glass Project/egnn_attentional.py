import torch
import torch.nn as nn
from torch_geometric.nn import MessagePassing
import math

class GaussianSmearing(nn.Module):
    def __init__(self, start=0.0, stop=4.0, num_gaussians=32):
        super().__init__()
        self.start = start
        self.stop = stop
        self.num_gaussians = num_gaussians
        offset = torch.linspace(start, stop, num_gaussians)
        self.coeff = -0.5 / (offset[1] - offset[0]).item()**2
        self.register_buffer('offset', offset)

    def forward(self, dist):
        dist = dist.view(-1, 1) - self.offset.view(1, -1)
        return torch.exp(self.coeff * torch.pow(dist, 2))

class MultiHeadAttentionalEGNNLayer(MessagePassing):
    def __init__(self, node_dim, edge_dim, vector_dim, heads=4):
        super().__init__(aggr='add')
        self.feat_dim = node_dim
        self.edge_dim = edge_dim
        self.vector_dim = vector_dim
        self.heads = heads
        
        # Distance Expansion
        self.distance_expansion = GaussianSmearing(start=0.0, stop=4.0, num_gaussians=edge_dim)

        # Message Function (Computes Multi-Head Attention weights & Values)
        # Input: h_i, h_j, RBF(dist)
        msg_in_dim = node_dim * 2 + edge_dim
        self.message_mlp = nn.Sequential(
            nn.Linear(msg_in_dim, node_dim),
            nn.SiLU(),
            nn.Linear(node_dim, node_dim * heads) # V
        )
        
        self.attention_mlp = nn.Sequential(
            nn.Linear(msg_in_dim, node_dim),
            nn.SiLU(),
            nn.Linear(node_dim, heads) # K attention logits
        )
        
        # Coordinate Update Gate
        # Output gating scalar for vectors (mux, muy) and coordinate displacement
        self.coord_mlp = nn.Sequential(
            nn.Linear(node_dim * heads, node_dim),
            nn.SiLU(),
            nn.Linear(node_dim, 1) # Scales displacement
        )
        
        self.vector_gate_mlp = nn.Sequential(
            nn.Linear(node_dim, node_dim),
            nn.SiLU(),
            nn.Linear(node_dim, 1) # Scales active self-propulsion
        )

        # Node Update Function
        self.node_mlp = nn.Sequential(
            nn.Linear(node_dim + node_dim * heads, node_dim),
            nn.SiLU(),
            nn.Linear(node_dim, node_dim)
        )

    def forward(self, h, pos, vec, edge_index):
        # h: [N, node_dim], pos: [N, 2], vec: [N, 2]
        return self.propagate(edge_index, h=h, x=pos, v=vec)

    def message(self, h_i, h_j, x_i, x_j, index, ptr, size_i):
        # Distance calculation
        diff = x_i - x_j
        dist = torch.norm(diff, dim=-1, keepdim=True) + 1e-8
        rbf_dist = self.distance_expansion(dist).squeeze(1) # [E, edge_dim]

        # Multi-Head Attention
        m_ij_input = torch.cat([h_i, h_j, rbf_dist], dim=-1)
        
        v_ij = self.message_mlp(m_ij_input).view(-1, self.heads, self.feat_dim)
        alpha_ij = self.attention_mlp(m_ij_input) # [E, heads]
        
        # Softmax over neighborhood
        from torch_geometric.utils import softmax
        alpha_ij = softmax(alpha_ij, index, ptr, size_i).unsqueeze(-1) # [E, heads, 1]
        
        # Attended message
        m_ij = (alpha_ij * v_ij).view(-1, self.heads * self.feat_dim) # [E, heads * node_dim]
        
        # Coordinate update msg
        coord_scale = self.coord_mlp(m_ij) # [E, 1]
        coord_msg = diff * coord_scale
        
        return m_ij, coord_msg

    def aggregate(self, inputs, index, ptr, dim_size):
        m_ij, coord_msg = inputs
        m_i = super().aggregate(m_ij, index, ptr, dim_size)
        coord_update = super().aggregate(coord_msg, index, ptr, dim_size)
        return m_i, coord_update

    def update(self, aggr_out, h, x, v):
        m_i, coord_update = aggr_out
        
        # Coordinate update (Vector Gated: combination of displacement + active vector)
        v_gate = self.vector_gate_mlp(h) # [N, 1]
        x_new = x + coord_update + v_gate * v
        
        # Node update with residual
        h_in = torch.cat([h, m_i], dim=-1)
        h_new = h + self.node_mlp(h_in)
        
        return h_new, x_new

class AttentionalEGNNModel(nn.Module):
    def __init__(self, scalar_in_dim=3, vector_dim=2, hidden_dim=64, edge_dim=32, num_layers=4, heads=4):
        super().__init__()
        self.embedding = nn.Linear(scalar_in_dim, hidden_dim)
        
        self.layers = nn.ModuleList([
            MultiHeadAttentionalEGNNLayer(hidden_dim, edge_dim, vector_dim, heads)
            for _ in range(num_layers)
        ])
        
        self.output_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.SiLU(),
            nn.Linear(hidden_dim // 2, 1)
        )

    def forward(self, data):
        # Extract features
        # Assuming data.x has [voronoi, ord1, ord2]
        # data.vec has [mux, muy]
        h = self.embedding(data.x)
        pos = data.pos
        vec = data.vec
        edge_index = data.edge_index
        
        for layer in self.layers:
            h, pos = layer(h, pos, vec, edge_index)
            
        out = self.output_head(h)
        return out.squeeze(-1)
