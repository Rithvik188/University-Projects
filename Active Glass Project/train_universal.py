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

def extract_domain_features(dump_file, f_act, target_t0):
    print(f"Loading graph from {dump_file} (Fact={f_act})...")
    pos0, pos50, box = parse_two_frames(dump_file, target_t0, target_t0 + 50000)
    Lx = box[1] - box[0]
    Ly = box[3] - box[2]
    
    N = len(pos0)
    vec = torch.zeros((N, 2), dtype=torch.float)
    x_feat = torch.zeros((N, 4), dtype=torch.float) # 4 features now!
    
    with open(dump_file, 'r') as fh:
        lines = fh.readlines()
        
    i = 0
    while i < len(lines):
        if lines[i].strip() == "ITEM: TIMESTEP":
            ts = int(lines[i+1].strip())
            if ts == target_t0:
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
                    x_feat[idx, 3] = f_act # Append F_act as 4th continuous feature
                    i += 1
                break
            else: i += 1
        else: i += 1
            
    # Standardize only first 3 features. 4th is constant for the domain
    for col in range(3):
        x_feat[:, col] = (x_feat[:, col] - x_feat[:, col].mean()) / (x_feat[:, col].std() + 1e-8)
        
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
    
    return pos, vec, x_feat, edge_index, y

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    # 1. Parse V1 and V2
    pos_v1, vec_v1, x_v1, edge_index_v1, y_v1 = extract_domain_features('trajectory_80k.dump', 1.0, TARGET_T0)
    pos_v2, vec_v2, x_v2, edge_index_v2, y_v2 = extract_domain_features('trajectory_80k_v2.dump', 1.25, TARGET_T0)
    
    N_v1 = len(pos_v1)
    N_v2 = len(pos_v2)
    N_total = N_v1 + N_v2
    
    print(f"Combining V1 ({N_v1}) and V2 ({N_v2}) into {N_total}-node disconnected super-graph...")
    
    pos = torch.cat([pos_v1, pos_v2])
    vec = torch.cat([vec_v1, vec_v2])
    x_feat = torch.cat([x_v1, x_v2])
    y = torch.cat([y_v1, y_v2])
    
    # Shift V2 edges
    edge_index_v2_shifted = edge_index_v2 + N_v1
    edge_index = torch.cat([edge_index_v1, edge_index_v2_shifted], dim=1)
    
    # 2. Setup Data and Random Mixed Mask
    data = Data(x=x_feat, pos=pos, vec=vec, edge_index=edge_index, y=y)
    
    indices = torch.randperm(N_total)
    train_size = int(0.8 * N_total)
    train_mask = torch.zeros(N_total, dtype=torch.bool)
    test_mask = torch.zeros(N_total, dtype=torch.bool)
    train_mask[indices[:train_size]] = True
    test_mask[indices[train_size:]] = True
    data.train_mask = train_mask
    data.test_mask = test_mask
    
    num_neighbors = [8, 4, 4, 4]
    sampler = PurePythonNeighborSampler(data.edge_index, num_neighbors, N_total)
    train_loader = NeighborLoader(data, num_neighbors=num_neighbors, batch_size=1024, input_nodes=data.train_mask, shuffle=True, neighbor_sampler=sampler)
    
    # We will just evaluate test manually for V1 and V2 separately to ensure accurate Pearson R split
    test_idx = indices[train_size:]
    test_idx_v1 = test_idx[test_idx < N_v1]
    test_idx_v2 = test_idx[test_idx >= N_v1]
    
    mask_v1 = torch.zeros(N_total, dtype=torch.bool)
    mask_v1[test_idx_v1] = True
    
    mask_v2 = torch.zeros(N_total, dtype=torch.bool)
    mask_v2[test_idx_v2] = True
    
    test_loader_v1 = NeighborLoader(data, num_neighbors=num_neighbors, batch_size=1024, input_nodes=mask_v1, shuffle=False, neighbor_sampler=sampler)
    test_loader_v2 = NeighborLoader(data, num_neighbors=num_neighbors, batch_size=1024, input_nodes=mask_v2, shuffle=False, neighbor_sampler=sampler)
    
    # 3. Model Initialization (FRESH WEIGHTS, scalar_in_dim=4)
    model = AttentionalEGNNModel(scalar_in_dim=4, vector_dim=2, hidden_dim=64, edge_dim=32, num_layers=4, heads=4).to(device)
    print("Initialized fresh Universal Attentional EGNN (scalar_in_dim=4).")
    
    # 4. Training
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    
    def weighted_mse_loss(pred, target):
        if len(target) > 0:
            threshold = torch.quantile(target, 0.85)
            weights = torch.where(target > threshold, torch.tensor(5.0, device=target.device), torch.tensor(1.0, device=target.device))
        else:
            weights = torch.ones_like(target)
        sq_err = (pred - target) ** 2
        return (weights * sq_err).mean()
    
    epochs = 1
    print(f"Training Universal Model on Mixed Data for {epochs} epoch...")
    
    model.train()
    for batch_idx, batch in enumerate(train_loader):
        batch = batch.to(device)
        optimizer.zero_grad()
        out = model(batch)
        loss = weighted_mse_loss(out[:batch.batch_size], batch.y[:batch.batch_size])
        loss.backward()
        optimizer.step()
        if batch_idx % 20 == 0:
            print(f"  Batch {batch_idx} | Mixed Loss: {loss.item():.4f}")
            
    # 5. Dual Verification
    model.eval()
    
    print("Evaluating on V1 Test Set...")
    preds_v1, targets_v1 = [], []
    with torch.no_grad():
        for batch in test_loader_v1:
            batch = batch.to(device)
            out = model(batch)
            preds_v1.extend(out[:batch.batch_size].cpu().numpy())
            targets_v1.extend(batch.y[:batch.batch_size].cpu().numpy())
    r_v1, _ = pearsonr(preds_v1, targets_v1)
    
    print("Evaluating on V2 Test Set...")
    preds_v2, targets_v2 = [], []
    with torch.no_grad():
        for batch in test_loader_v2:
            batch = batch.to(device)
            out = model(batch)
            preds_v2.extend(out[:batch.batch_size].cpu().numpy())
            targets_v2.extend(batch.y[:batch.batch_size].cpu().numpy())
    r_v2, _ = pearsonr(preds_v2, targets_v2)
    
    print("\n" + "="*40)
    print("UNIVERSAL MODEL DUAL VERIFICATION RESULTS")
    print(f"V1 (Fact=1.00) Test Pearson R: {r_v1:.4f}")
    print(f"V2 (Fact=1.25) Test Pearson R: {r_v2:.4f}")
    print("="*40)
    
    torch.save(model.state_dict(), 'egnn_universal_model.pth')
    print("Saved egnn_universal_model.pth")
    
if __name__ == "__main__":
    main()
