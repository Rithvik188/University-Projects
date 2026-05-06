import os
import torch
import torch.nn as nn
import numpy as np
from scipy.spatial import cKDTree
from scipy.stats import pearsonr
from torch_geometric.data import Data
from torch_geometric.loader import NeighborLoader
import random
from torch_geometric.sampler.base import BaseSampler, SamplerOutput

from egnn_attentional import AttentionalEGNNModel
from prepare_80k_labels import parse_two_frames, TARGET_T0, TARGET_T50, SIGMA, R_SMOOTH, spatial_gaussian_smooth

class PurePythonNeighborSampler(BaseSampler):
    def __init__(self, edge_index, num_neighbors, num_nodes):
        self.num_neighbors = num_neighbors
        self.num_nodes = num_nodes
        self.adj = [[] for _ in range(num_nodes)]
        src = edge_index[0].numpy()
        dst = edge_index[1].numpy()
        for e_idx in range(len(src)):
            u, v = src[e_idx], dst[e_idx]
            self.adj[v].append((u, e_idx))
            
    def sample_from_nodes(self, inputs):
        seed_nodes = inputs.node.tolist()
        nodes = list(seed_nodes)
        node_to_idx = {n: i for i, n in enumerate(nodes)}
        row_list, col_list, edge_list = [], [], []
        current_layer_nodes = seed_nodes
        
        for k in self.num_neighbors:
            next_layer_nodes = []
            for v in current_layer_nodes:
                neighbors = self.adj[v]
                if len(neighbors) == 0: continue
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
            if len(current_layer_nodes) == 0: break
                
        return SamplerOutput(
            node=torch.tensor(nodes, dtype=torch.long),
            row=torch.tensor(row_list, dtype=torch.long),
            col=torch.tensor(col_list, dtype=torch.long),
            edge=torch.tensor(edge_list, dtype=torch.long),
            batch=inputs.input_id,
            metadata=(inputs.input_id, inputs.time)
        )
        
    def sample_from_edges(self, inputs):
        raise NotImplementedError()

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    # 1. Parse V2 and compute log labels on the fly
    DUMP_FILE = 'trajectory_80k_v2.dump'
    print(f"Loading V2 graph from {DUMP_FILE}...")
    pos0, pos50, box = parse_two_frames(DUMP_FILE, TARGET_T0, TARGET_T50)
    Lx = box[1] - box[0]
    Ly = box[3] - box[2]
    
    N = len(pos0)
    vec = torch.zeros((N, 2), dtype=torch.float)
    x_feat = torch.zeros((N, 3), dtype=torch.float)
    
    with open(DUMP_FILE, 'r') as fh:
        lines = fh.readlines()
        
    i = 0
    while i < len(lines):
        if lines[i].strip() == "ITEM: TIMESTEP":
            ts = int(lines[i+1].strip())
            if ts == TARGET_T0:
                num_atoms = int(lines[i+3].strip())
                i += 9
                for _ in range(num_atoms):
                    parts = lines[i].split()
                    idx = int(parts[0]) - 1
                    vec[idx, 0] = float(parts[4])
                    vec[idx, 1] = float(parts[5])
                    x_feat[idx, 0] = float(parts[6])
                    x_feat[idx, 1] = float(parts[7])
                    x_feat[idx, 2] = float(parts[8])
                    i += 1
                break
            else: i += 1
        else: i += 1
            
    x_feat = (x_feat - x_feat.mean(dim=0, keepdim=True)) / (x_feat.std(dim=0, keepdim=True) + 1e-8)
    
    delta = pos50 - pos0
    delta[:, 0] -= Lx * np.round(delta[:, 0] / Lx)
    delta[:, 1] -= Ly * np.round(delta[:, 1] / Ly)
    sq_disp = (delta ** 2).sum(axis=1)
    
    gt_smoothed = spatial_gaussian_smooth(pos0, sq_disp, SIGMA, R_SMOOTH)
    gt_log = np.log(gt_smoothed + 1e-8)
    
    pos = torch.tensor(pos0, dtype=torch.float)
    tree = cKDTree(pos.numpy())
    pairs = tree.query_pairs(r=4.0, output_type='ndarray')
    edges_ij = pairs.T
    edges_ji = np.vstack([edges_ij[1], edges_ij[0]])
    edge_index = torch.tensor(np.hstack([edges_ij, edges_ji]), dtype=torch.long)
    y = torch.tensor(gt_log, dtype=torch.float)
    
    data = Data(x=x_feat, pos=pos, vec=vec, edge_index=edge_index, y=y)
    
    # Setup masks
    indices = torch.randperm(N)
    train_size = int(0.8 * N)
    train_mask = torch.zeros(N, dtype=torch.bool)
    test_mask = torch.zeros(N, dtype=torch.bool)
    train_mask[indices[:train_size]] = True
    test_mask[indices[train_size:]] = True
    data.train_mask = train_mask
    data.test_mask = test_mask
    
    num_neighbors = [8, 4, 4, 4]
    sampler = PurePythonNeighborSampler(data.edge_index, num_neighbors, N)
    train_loader = NeighborLoader(data, num_neighbors=num_neighbors, batch_size=1024, input_nodes=data.train_mask, shuffle=True, neighbor_sampler=sampler)
    test_loader = NeighborLoader(data, num_neighbors=num_neighbors, batch_size=1024, input_nodes=data.test_mask, shuffle=False, neighbor_sampler=sampler)
    
    # 2. Load V1 weights
    model = AttentionalEGNNModel(scalar_in_dim=3, vector_dim=2, hidden_dim=64, edge_dim=32, num_layers=4, heads=4).to(device)
    model.load_state_dict(torch.load('egnn_attentional_model_80k.pth', map_location=device))
    print("Pre-trained V1 weights loaded successfully.")
    
    # 3. Fine-Tune with low learning rate
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4) # LOW LR
    
    def weighted_mse_loss(pred, target):
        if len(target) > 0:
            threshold = torch.quantile(target, 0.85)
            weights = torch.where(target > threshold, torch.tensor(5.0, device=target.device), torch.tensor(1.0, device=target.device))
        else:
            weights = torch.ones_like(target)
        sq_err = (pred - target) ** 2
        return (weights * sq_err).mean()
    
    criterion = weighted_mse_loss
    
    epochs = 1
    print(f"Fine-tuning on V2 data for {epochs} epoch...")
    
    model.train()
    for batch_idx, batch in enumerate(train_loader):
        batch = batch.to(device)
        optimizer.zero_grad()
        out = model(batch)
        loss = criterion(out[:batch.batch_size], batch.y[:batch.batch_size])
        loss.backward()
        optimizer.step()
        if batch_idx % 10 == 0:
            print(f"  Batch {batch_idx} | Fine-Tune Loss: {loss.item():.4f}")
            
    # Eval
    model.eval()
    all_preds, all_targets = [], []
    with torch.no_grad():
        for batch in test_loader:
            batch = batch.to(device)
            out = model(batch)
            all_preds.extend(out[:batch.batch_size].cpu().numpy())
            all_targets.extend(batch.y[:batch.batch_size].cpu().numpy())
            
    pearson_r, _ = pearsonr(all_preds, all_targets)
    print(f"Fine-Tuned V2 Pearson R: {pearson_r:.4f}")
    
    # Save the Fine-Tuned Model
    torch.save(model.state_dict(), 'egnn_attentional_model_80k_v2.pth')
    print("Saved egnn_attentional_model_80k_v2.pth")
    
if __name__ == "__main__":
    main()
