import os
import random
import torch
import torch.nn as nn
import numpy as np
from scipy.spatial import cKDTree
from scipy.stats import pearsonr
from torch_geometric.data import Data
from torch_geometric.loader import NeighborLoader
from torch_geometric.sampler.base import BaseSampler, SamplerOutput

from egnn_attentional import AttentionalEGNNModel
from prepare_80k_labels import parse_two_frames, TARGET_T0, TARGET_T50

class PurePythonNeighborSampler(BaseSampler):
    """
    A pure Python neighbor sampler to bypass the pyg-lib/torch-sparse
    C++ extension requirements on Windows for NeighborLoader.
    """
    def __init__(self, edge_index, num_neighbors, num_nodes):
        self.num_neighbors = num_neighbors
        self.num_nodes = num_nodes
        
        print("  [Sampler] Building adjacency list for Python sampler...")
        self.adj = [[] for _ in range(num_nodes)]
        # edge_index is [2, E] -> source, target
        # PyG undirected graphs have both directions.
        src = edge_index[0].numpy()
        dst = edge_index[1].numpy()
        for e_idx in range(len(src)):
            u, v = src[e_idx], dst[e_idx]
            self.adj[v].append((u, e_idx))
            
        print("  [Sampler] Adjacency list built.")
            
    def sample_from_nodes(self, inputs):
        seed_nodes = inputs.node.tolist()
        
        nodes = list(seed_nodes)
        node_to_idx = {n: i for i, n in enumerate(nodes)}
        
        row_list = []
        col_list = []
        edge_list = []
        
        current_layer_nodes = seed_nodes
        
        for k in self.num_neighbors:
            next_layer_nodes = []
            for v in current_layer_nodes:
                neighbors = self.adj[v]
                if len(neighbors) == 0:
                    continue
                    
                if k > 0 and len(neighbors) > k:
                    sampled = random.sample(neighbors, k)
                else:
                    sampled = neighbors
                    
                for u, e_idx in sampled:
                    if u not in node_to_idx:
                        node_to_idx[u] = len(nodes)
                        nodes.append(u)
                        next_layer_nodes.append(u)
                        
                    row_list.append(node_to_idx[u])
                    col_list.append(node_to_idx[v])
                    edge_list.append(e_idx)
                    
            current_layer_nodes = next_layer_nodes
            if len(current_layer_nodes) == 0:
                break
                
        return SamplerOutput(
            node=torch.tensor(nodes, dtype=torch.long),
            row=torch.tensor(row_list, dtype=torch.long),
            col=torch.tensor(col_list, dtype=torch.long),
            edge=torch.tensor(edge_list, dtype=torch.long),
            batch=inputs.input_id,
            metadata=(inputs.input_id, inputs.time)
        )
        
    def sample_from_edges(self, inputs):
        raise NotImplementedError("sample_from_edges not needed for this script.")

def load_80k_graph():
    print("Loading 80k graph at T=0...")
    # parse the dump file using the robust parser
    pos0, _, _ = parse_two_frames('trajectory_80k.dump', TARGET_T0, TARGET_T50)
    
    pos = torch.tensor(pos0, dtype=torch.float)
    
    # Random dummy features since we only extracted pos in parse_two_frames,
    # but wait, the prompt says "Train our Attentional EGNN exclusively on this first 50k-step window."
    # We should parse the actual features for T=0! Let's re-parse or use dummy if acceptable.
    # Actually, we can read the features from the dump for T=0.
    print("Extracting features from dump...")
    N = 80000
    vec = torch.zeros((N, 2), dtype=torch.float)
    x_feat = torch.zeros((N, 3), dtype=torch.float)
    
    with open('trajectory_80k.dump', 'r') as fh:
        lines = fh.readlines()
    
    i = 0
    while i < len(lines):
        if lines[i].strip() == "ITEM: TIMESTEP":
            ts = int(lines[i+1].strip())
            if ts == TARGET_T0:
                # read this block
                num_atoms = int(lines[i+3].strip())
                i += 9 # skip headers
                for _ in range(num_atoms):
                    parts = lines[i].split()
                    idx = int(parts[0]) - 1 # 1-indexed to 0-indexed assuming 1 to 80000
                    vec[idx, 0] = float(parts[4])
                    vec[idx, 1] = float(parts[5])
                    x_feat[idx, 0] = float(parts[6])
                    x_feat[idx, 1] = float(parts[7])
                    x_feat[idx, 2] = float(parts[8])
                    i += 1
                break
            else:
                i += 1
        else:
            i += 1
            
    # Standardize
    mean_feat = x_feat.mean(dim=0, keepdim=True)
    std_feat = x_feat.std(dim=0, keepdim=True) + 1e-8
    x_feat = (x_feat - mean_feat) / std_feat
    
    print("Building cKDTree (rc=4.0) on 80k nodes...")
    tree = cKDTree(pos.numpy())
    pairs = tree.query_pairs(r=4.0, output_type='ndarray')
    edges_ij = pairs.T
    edges_ji = np.vstack([edges_ij[1], edges_ij[0]])
    edge_index = torch.tensor(np.hstack([edges_ij, edges_ji]), dtype=torch.long)
    
    labels = np.load('labels_80k.npy')
    y = torch.tensor(labels, dtype=torch.float)
    
    data = Data(x=x_feat, pos=pos, vec=vec, edge_index=edge_index, y=y)
    return data

def train():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    data = load_80k_graph()
    
    num_nodes = data.num_nodes
    indices = torch.randperm(num_nodes)
    train_size = int(0.8 * num_nodes)
    
    train_mask = torch.zeros(num_nodes, dtype=torch.bool)
    test_mask = torch.zeros(num_nodes, dtype=torch.bool)
    train_mask[indices[:train_size]] = True
    test_mask[indices[train_size:]] = True
    
    data.train_mask = train_mask
    data.test_mask = test_mask
    
    print("Creating custom Sampler and NeighborLoader (Batch Size: 1024)...")
    num_neighbors = [8, 4, 4, 4]
    sampler = PurePythonNeighborSampler(data.edge_index, num_neighbors, num_nodes)
    
    train_loader = NeighborLoader(
        data,
        num_neighbors=num_neighbors,
        batch_size=1024,
        input_nodes=data.train_mask,
        shuffle=True,
        neighbor_sampler=sampler
    )
    
    test_loader = NeighborLoader(
        data,
        num_neighbors=num_neighbors,
        batch_size=1024,
        input_nodes=data.test_mask,
        shuffle=False,
        neighbor_sampler=sampler
    )
    
    model = AttentionalEGNNModel(
        scalar_in_dim=3, vector_dim=2, hidden_dim=64, edge_dim=32, num_layers=4, heads=4
    ).to(device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    
    def weighted_mse_loss(pred, target):
        # Dynamically compute the 85th percentile threshold for this batch
        if len(target) > 0:
            threshold = torch.quantile(target, 0.85)
            weights = torch.where(target > threshold, torch.tensor(5.0, device=target.device), torch.tensor(1.0, device=target.device))
        else:
            weights = torch.ones_like(target)
        sq_err = (pred - target) ** 2
        return (weights * sq_err).mean()
    
    criterion = weighted_mse_loss
    
    epochs = 1
    print(f"Starting training on 80k for {epochs} epochs...")
    
    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0
        
        for batch_idx, batch in enumerate(train_loader):
            batch = batch.to(device)
            optimizer.zero_grad()
            
            out = model(batch)
            loss = criterion(out[:batch.batch_size], batch.y[:batch.batch_size])
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item() * batch.batch_size
            if batch_idx % 10 == 0:
                print(f"  Epoch {epoch} | Batch {batch_idx} | Loss: {loss.item():.4f}")
            
        avg_train_loss = total_loss / train_size
        
        model.eval()
        all_preds, all_targets = [], []
        with torch.no_grad():
            for batch in test_loader:
                batch = batch.to(device)
                out = model(batch)
                preds = out[:batch.batch_size].cpu().numpy()
                targets = batch.y[:batch.batch_size].cpu().numpy()
                all_preds.extend(preds)
                all_targets.extend(targets)
                
        pearson_r, _ = pearsonr(all_preds, all_targets)
        print(f"Epoch {epoch}/{epochs} | Train MSE: {avg_train_loss:.4f} | Test Pearson R: {pearson_r:.4f}")
        
    print(f"Final Pearson R correlation: {pearson_r:.4f}")
    
    # Save the new weights
    torch.save(model.state_dict(), 'egnn_attentional_model_80k.pth')
    print("Saved egnn_attentional_model_80k.pth")
    
if __name__ == "__main__":
    train()
