import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
from scipy.spatial import cKDTree
from torch_geometric.data import Data
import os

from egnn_attentional import AttentionalEGNNModel

def load_lammps_dump(dump_path):
    """
    Parses LAMMPS dump file and returns PyG Data object.
    Schema expected: id type x y mux muy c_voronoi c_ord[1] c_ord[2]
    """
    print(f"Loading data from {dump_path}...")
    atoms = []
    
    with open(dump_path, 'r') as f:
        lines = f.readlines()
        
    in_atoms = False
    for line in lines:
        if line.startswith("ITEM: ATOMS"):
            in_atoms = True
            continue
        elif line.startswith("ITEM: TIMESTEP") and in_atoms:
            # We only parse the first frame for this dataset
            break
            
        if in_atoms:
            parts = line.strip().split()
            # 0:id, 1:type, 2:x, 3:y, 4:mux, 5:muy, 6:c_voronoi, 7:c_ord1, 8:c_ord2
            atom_id = int(parts[0])
            x, y = float(parts[2]), float(parts[3])
            mux, muy = float(parts[4]), float(parts[5])
            vor = float(parts[6])
            ord1, ord2 = float(parts[7]), float(parts[8])
            atoms.append({
                'id': atom_id, 'x': x, 'y': y,
                'mux': mux, 'muy': muy,
                'vor': vor, 'ord1': ord1, 'ord2': ord2
            })
            
    # Sort by ID to ensure alignment with labels
    atoms.sort(key=lambda a: a['id'])
    
    pos = torch.tensor([[a['x'], a['y']] for a in atoms], dtype=torch.float)
    vec = torch.tensor([[a['mux'], a['muy']] for a in atoms], dtype=torch.float)
    x_feat = torch.tensor([[a['vor'], a['ord1'], a['ord2']] for a in atoms], dtype=torch.float)
    
    # Standardize features
    mean_feat = x_feat.mean(dim=0, keepdim=True)
    std_feat = x_feat.std(dim=0, keepdim=True) + 1e-8
    x_feat = (x_feat - mean_feat) / std_feat
    
    # Construct Radius Graph
    print("Constructing radius graph (rc = 4.0)...")
    tree = cKDTree(pos.numpy())
    pairs = tree.query_pairs(r=4.0, output_type='ndarray')
    
    if len(pairs) > 0:
        edges_ij = pairs.T
        edges_ji = np.vstack([edges_ij[1], edges_ij[0]])
        edge_index = torch.tensor(np.hstack([edges_ij, edges_ji]), dtype=torch.long)
    else:
        edge_index = torch.empty((2, 0), dtype=torch.long)
        
    data = Data(x=x_feat, pos=pos, vec=vec, edge_index=edge_index)
    return data

def train():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    dump_file = 'trajectory_2k.dump'
    labels_file = 'labels.npy'
    
    # Allow running with mock data if files don't exist yet for verification
    if not os.path.exists(dump_file):
        print("Warning: dump file not found, generating mock data for testing logic.")
        N = 2000
        pos = torch.rand((N, 2)) * 68.0
        vec = torch.randn((N, 2))
        x_feat = torch.randn((N, 3))
        
        tree = cKDTree(pos.numpy())
        pairs = tree.query_pairs(r=4.0, output_type='ndarray')
        edges_ij = pairs.T
        edges_ji = np.vstack([edges_ij[1], edges_ij[0]])
        edge_index = torch.tensor(np.hstack([edges_ij, edges_ji]), dtype=torch.long)
        
        data = Data(x=x_feat, pos=pos, vec=vec, edge_index=edge_index)
        y = torch.randn(N)
    else:
        data = load_lammps_dump(dump_file)
        y_np = np.load(labels_file)
        
        # In case the labels file is for the full 80k system but we sliced 2k
        if len(y_np) > data.num_nodes:
            print(f"Labels length {len(y_np)} > nodes {data.num_nodes}. Assuming first {data.num_nodes} matches the slice or labels correspond to ordered IDs.")
            # We will just take the first N labels if it's a test slice
            y = torch.tensor(y_np[:data.num_nodes], dtype=torch.float)
        else:
            y = torch.tensor(y_np, dtype=torch.float)
            
    data.y = y
    
    # Train/Test Split (Node level)
    num_nodes = data.num_nodes
    indices = torch.randperm(num_nodes)
    train_size = int(0.8 * num_nodes)
    
    train_mask = torch.zeros(num_nodes, dtype=torch.bool)
    test_mask = torch.zeros(num_nodes, dtype=torch.bool)
    train_mask[indices[:train_size]] = True
    test_mask[indices[train_size:]] = True
    
    data.train_mask = train_mask
    data.test_mask = test_mask
    
    # Full batch training since NeighborLoader requires torch-sparse on Windows
    data = data.to(device)
    
    model = AttentionalEGNNModel(
        scalar_in_dim=3,
        vector_dim=2,
        hidden_dim=64,
        edge_dim=32,
        num_layers=4,
        heads=4
    ).to(device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
    criterion = nn.MSELoss()
    
    epochs = 100
    print("Starting training...")
    
    for epoch in range(1, epochs + 1):
        model.train()
        optimizer.zero_grad()
        
        out = model(data)
        loss = criterion(out[data.train_mask], data.y[data.train_mask])
        loss.backward()
        optimizer.step()
        
        avg_train_loss = loss.item()
        
        # Evaluation
        if epoch % 10 == 0 or epoch == epochs:
            model.eval()
            with torch.no_grad():
                out = model(data)
                preds = out[data.test_mask].cpu().numpy()
                targets = data.y[data.test_mask].cpu().numpy()
                
            pearson_r, _ = pearsonr(preds, targets)
            
            print(f"Epoch {epoch:03d}/{epochs} | Train MSE: {avg_train_loss:.4f} | Test Pearson R: {pearson_r:.4f}")

    print("Training complete. Saving model...")
    torch.save(model.state_dict(), 'egnn_attentional_model.pth')

if __name__ == "__main__":
    train()
