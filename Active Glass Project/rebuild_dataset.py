"""
Rebuilds the glass graph dataset with correct per-node labels.
labels.npy is (1024,) — one propensity value per particle, same for all frames.
Each graph (frame) gets y = full (1024,) tensor of per-node propensity labels.
"""
import numpy as np
import torch
from torch_geometric.data import Data
from scipy.spatial import cKDTree

def rebuild_dataset():
    print("Loading raw data...")
    trajectory = np.load('trajectory.npy')  # (200, 1024, 2)
    features = np.load('features.npy')      # (200, 1024, 2)
    labels = np.load('labels.npy')          # (1024,) — per-particle propensity

    num_frames, num_particles, _ = trajectory.shape
    print(f"Frames: {num_frames}, Particles: {num_particles}")
    print(f"Labels shape: {labels.shape} (per-particle, same for all frames)")

    # Normalize features globally
    mean_feat = features.mean(axis=(0, 1), keepdims=True)
    std_feat = features.std(axis=(0, 1), keepdims=True)
    features_norm = (features - mean_feat) / (std_feat + 1e-8)

    # Build per-node label tensor (shared across all frames)
    y_tensor = torch.tensor(labels, dtype=torch.float)  # (1024,)

    dataset = []
    print("Rebuilding graphs with per-node labels...")
    for i in range(num_frames):
        pos = torch.tensor(trajectory[i], dtype=torch.float)
        x = torch.tensor(features_norm[i], dtype=torch.float)

        # Build radius graph (r_cut = 2.0)
        tree = cKDTree(pos.numpy())
        pairs = tree.query_pairs(r=2.0, output_type='ndarray')
        if len(pairs) > 0:
            edges_ij = pairs.T
            edges_ji = np.vstack([edges_ij[1], edges_ij[0]])
            edge_index_np = np.hstack([edges_ij, edges_ji])
            edge_index = torch.tensor(edge_index_np, dtype=torch.long)
        else:
            edge_index = torch.empty((2, 0), dtype=torch.long)

        # y is per-node: shape (1024,)
        data = Data(x=x, edge_index=edge_index, pos=pos, y=y_tensor)
        dataset.append(data)

        if (i + 1) % 50 == 0:
            print(f"  Processed {i+1}/{num_frames} frames")

    torch.save(dataset, 'glass_graph_data.pt')
    print("\nDataset rebuilt and saved to glass_graph_data.pt")
    print(f"Sample: x={dataset[0].x.shape}, y={dataset[0].y.shape}, pos={dataset[0].pos.shape}")

if __name__ == '__main__':
    rebuild_dataset()
