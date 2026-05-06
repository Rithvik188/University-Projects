import torch
import torch.nn as nn
from torch_geometric.nn import MessagePassing

class EGNNLayer(MessagePassing):
    """
    E(n)-Equivariant Graph Neural Network Layer.
    Implementation inspired by Satorras et al.
    """
    def __init__(self, hidden_dim=64):
        # We use standard summation for aggregation over neighbors
        super(EGNNLayer, self).__init__(aggr='add')
        
        # Message Function MLP: phi_e
        # Input: h_i, h_j, squared distance (hidden_dim + hidden_dim + 1)
        self.edge_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 2 + 1, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU()
        )
        
        # Coordinate Update MLP: phi_x
        # Input: message m_ij (hidden_dim) -> Output: 1 scalar multiplier
        # Notice how this scales the displacement vector (x_i - x_j)
        self.coord_mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 1)
        )
        
        # Node Update MLP: phi_h
        # Input: h_i, aggregated message (hidden_dim + hidden_dim)
        self.node_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
    def forward(self, h, pos, edge_index):
        # Propagate calls message(), aggregate(), update()
        return self.propagate(edge_index, h=h, x=pos)

    def message(self, h_i, h_j, x_i, x_j):
        # Compute squared Euclidean distance ||x_i - x_j||^2
        sq_dist = torch.sum((x_i - x_j) ** 2, dim=-1, keepdim=True)
        
        # Concatenate h_i, h_j, and distance to form input for message function
        m_ij_input = torch.cat([h_i, h_j, sq_dist], dim=-1)
        
        # Compute message m_ij = phi_e(h_i, h_j, sq_dist)
        m_ij = self.edge_mlp(m_ij_input)
        
        # Prepare components for coordinate update during aggregation
        # (x_i - x_j) * phi_x(m_ij)
        coord_msg = (x_i - x_j) * self.coord_mlp(m_ij)
        
        return m_ij, coord_msg

    def aggregate(self, inputs, index, dim_size=None):
        m_ij, coord_msg = inputs
        
        # Message aggregation: sum(m_ij)
        m_i = super().aggregate(m_ij, index, dim_size=dim_size)
        
        # Coordinate aggregation: sum((x_i - x_j) * phi_x(m_ij))
        coord_update = super().aggregate(coord_msg, index, dim_size=dim_size)
        
        return m_i, coord_update

    def update(self, aggr_out, h, x):
        m_i, coord_update = aggr_out
        
        # Coordinate Update: x_i = x_i + sum(...)
        x_new = x + coord_update
        
        # Node Update: h_i = phi_h(h_i, sum(m_ij))
        h_input = torch.cat([h, m_i], dim=-1)
        h_new = self.node_mlp(h_input)
        
        # Keep residual connection for stable node feature updates
        h_new = h + h_new
        
        return h_new, x_new


class EGNNModel(nn.Module):
    """
    Implementation of the E(n)-Equivariant Graph Neural Network.
    Designed for 4 layers, hidden dim 64, returning a single scalar.
    """
    def __init__(self, in_features=2, hidden_dim=64, num_layers=4):
        super(EGNNModel, self).__init__()
        
        # Expand original 2 input features to the hidden dimension
        self.embedding = nn.Linear(in_features, hidden_dim)
        
        # Stack of EGNN Message Passing Layers
        self.layers = nn.ModuleList([
            EGNNLayer(hidden_dim=hidden_dim) for _ in range(num_layers)
        ])
        
        # Final Node-level representation to predict the scalar propensity parameter
        self.output_mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 1) # Output predicted propensity
        )
        
    def forward(self, data):
        # Extract features and structure from the PyG Data object
        h = data.x           # Shape: (N, 2)
        pos = data.pos       # Shape: (N, 2)
        edge_index = data.edge_index
        
        # 1. Embed node features
        h = self.embedding(h)
        
        # 2. Iterate through Equivariant Layers
        for layer in self.layers:
            h, pos = layer(h, pos, edge_index)
            
        # 3. Predict Single Number Output (Propensity) form Node Representation
        out = self.output_mlp(h)
        
        return out.squeeze() # Remove explicit channel dim of 1
