import numpy as np
import torch
from torch_geometric.data import Data
from scipy.spatial import cKDTree
import matplotlib.pyplot as plt

def create_dataset():
    print("Loading data...")
    # Load data
    trajectory = np.load('trajectory.npy')  # positions
    features = np.load('features.npy')      # Voronoi and q6
    labels = np.load('labels.npy')          # Propensity

    num_frames = trajectory.shape[0]
    num_particles = trajectory.shape[1]
    print(f"Frames: {num_frames}, Particles per frame: {num_particles}")

    # Normalize Voronoi area and q6
    mean_feat = features.mean(axis=(0, 1), keepdims=True)
    std_feat = features.std(axis=(0, 1), keepdims=True)
    features_normalized = (features - mean_feat) / (std_feat + 1e-8)

    dataset = []

    print("Constructing graphs...")
    for i in range(num_frames):
        pos = torch.tensor(trajectory[i], dtype=torch.float)
        x = torch.tensor(features_normalized[i], dtype=torch.float)
        y = torch.tensor(labels[i], dtype=torch.float)

        # Define edges using a Radius Graph with r_cut = 2.0
        # PyG's radius_graph requires torch-cluster, which failed to install.
        # Using scipy.spatial.cKDTree instead.
        tree = cKDTree(pos.numpy())
        pairs = tree.query_pairs(r=2.0, output_type='ndarray')
        if len(pairs) > 0:
            edges_ij = pairs.T
            edges_ji = np.vstack([edges_ij[1], edges_ij[0]])
            edge_index_np = np.hstack([edges_ij, edges_ji])
            edge_index = torch.tensor(edge_index_np, dtype=torch.long)
        else:
            edge_index = torch.empty((2, 0), dtype=torch.long)

        # Create Data object
        data = Data(x=x, edge_index=edge_index, pos=pos, y=y)
        dataset.append(data)

        # For visualization, pick the first frame and a single particle (e.g., node 500)
        if i == 0:
            visualize_graph(pos, edge_index, target_node=500)

    # Save dataset
    print("Saving dataset...")
    torch.save(dataset, 'glass_graph_data.pt')
    print("Saved 'glass_graph_data.pt'")

    # Verification
    print(f"Number of graphs matches frames ({num_frames}): {len(dataset) == num_frames}")
    nodes_correct = all(data.num_nodes == 1024 for data in dataset)
    print(f"All graphs have 1024 nodes: {nodes_correct}")
    if nodes_correct and len(dataset) == num_frames:
        print("Verification successful!")
    else:
        print("Verification failed!")

def visualize_graph(pos, edge_index, target_node=0):
    print(f"Generating graph visualization for node {target_node}...")
    plt.figure(figsize=(8, 8))
    pos_np = pos.numpy()
    
    # Plot all particles faintly
    plt.scatter(pos_np[:, 0], pos_np[:, 1], s=10, c='lightgray', alpha=0.5, label='Other particles')

    # Find neighbors of target_node
    mask = edge_index[0] == target_node
    neighbors = edge_index[1][mask].numpy()

    # Draw edges
    for neighbor in neighbors:
        x_vals = [pos_np[target_node, 0], pos_np[neighbor, 0]]
        y_vals = [pos_np[target_node, 1], pos_np[neighbor, 1]]
        plt.plot(x_vals, y_vals, 'k-', alpha=0.6, linewidth=1)

    # Highlight target node and neighbors
    plt.scatter(pos_np[target_node, 0], pos_np[target_node, 1], s=80, c='red', edgecolors='k', zorder=5, label='Target Particle')
    if len(neighbors) > 0:
        plt.scatter(pos_np[neighbors, 0], pos_np[neighbors, 1], s=50, c='royalblue', edgecolors='k', zorder=4, label='Neighbors ($r < 2.0$)')

    plt.title(f'Radius Graph Connectivity (Particle {target_node}, $r_{{cut}}=2.0$)')
    # Use standard plot limits to zoom in or show full system. Showing full system with zoom.
    # We can zoom in roughly around the particle
    plt.xlim(pos_np[target_node, 0] - 5, pos_np[target_node, 0] + 5)
    plt.ylim(pos_np[target_node, 1] - 5, pos_np[target_node, 1] + 5)
    
    plt.legend()
    plt.tight_layout()
    plt.savefig('graph_sample.png', dpi=300)
    plt.close()
    print("Saved 'graph_sample.png'")

if __name__ == "__main__":
    create_dataset()
