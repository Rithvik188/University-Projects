import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import matplotlib.tri as mtri
from scipy.spatial import cKDTree
from torch_geometric.data import Data
import os

from egnn_attentional import AttentionalEGNNModel

def parse_dump_all_frames(dump_path):
    """Parses a LAMMPS dump file with multiple frames."""
    print(f"Parsing full trajectory from {dump_path}...")
    frames = []

    with open(dump_path, 'r') as f:
        lines = f.readlines()

    i = 0
    while i < len(lines):
        line = lines[i].strip()

        if line == "ITEM: TIMESTEP":
            timestep = int(lines[i + 1].strip())
            i += 2
        elif line == "ITEM: NUMBER OF ATOMS":
            num_atoms = int(lines[i + 1].strip())
            i += 2
        elif line.startswith("ITEM: BOX BOUNDS"):
            xlo, xhi = map(float, lines[i + 1].split()[:2])
            ylo, yhi = map(float, lines[i + 2].split()[:2])
            # skip z bounds line
            i += 4
        elif line.startswith("ITEM: ATOMS"):
            i += 1  # skip the "ITEM: ATOMS ..." header line
            atoms = []
            for _ in range(num_atoms):
                parts = lines[i].strip().split()
                atoms.append({
                    'id': int(parts[0]),
                    'x': float(parts[2]), 'y': float(parts[3]),
                    'mux': float(parts[4]), 'muy': float(parts[5]),
                    'vor': float(parts[6]), 'ord1': float(parts[7]), 'ord2': float(parts[8])
                })
                i += 1
            atoms.sort(key=lambda a: a['id'])
            frames.append({
                'timestep': timestep,
                'atoms': atoms,
                'box': (xlo, xhi, ylo, yhi)
            })
        else:
            i += 1

    print(f"Found {len(frames)} frame(s).")
    return frames

def build_data_from_frame(frame_data):
    atoms = frame_data['atoms']
    pos = torch.tensor([[a['x'], a['y']] for a in atoms], dtype=torch.float)
    vec = torch.tensor([[a['mux'], a['muy']] for a in atoms], dtype=torch.float)
    x_feat = torch.tensor([[a['vor'], a['ord1'], a['ord2']] for a in atoms], dtype=torch.float)
    
    # Standardize
    mean_feat = x_feat.mean(dim=0, keepdim=True)
    std_feat = x_feat.std(dim=0, keepdim=True) + 1e-8
    x_feat = (x_feat - mean_feat) / std_feat
    
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

def create_softness_map_animation(dump_path, model_path='egnn_attentional_model.pth', output_mp4='propensity_animation.mp4'):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Generating animation on {device}...")
    
    if not os.path.exists(dump_path):
        print(f"Error: {dump_path} not found. Ensure simulation has completed.")
        return
        
    frames = parse_dump_all_frames(dump_path)
    if not frames:
        print("No frames found in dump.")
        return
        
    # Load model
    model = AttentionalEGNNModel(
        scalar_in_dim=3, vector_dim=2, hidden_dim=64, edge_dim=32, num_layers=4, heads=4
    ).to(device)
    
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
        print("Loaded trained model weights.")
    else:
        print("Warning: Model weights not found. Using untrained model for animation.")
        
    model.eval()
    
    # Pre-compute all predictions to make animation fast
    print("Running inference on all frames...")
    predictions = []
    
    for f_idx, frame in enumerate(frames):
        data = build_data_from_frame(frame)
        num_nodes = data.num_nodes
        
        # Full-batch inference (no NeighborLoader needed)
        data_dev = data.to(device)
        with torch.no_grad():
            out = model(data_dev)
            frame_preds = out.cpu().numpy()
                
        predictions.append(frame_preds)
        print(f"Processed frame {f_idx+1}/{len(frames)}")

    # Create Animation Plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    plt.tight_layout(pad=4.0)
    
    def update(frame_idx):
        ax1.clear()
        ax2.clear()
        
        frame = frames[frame_idx]
        preds = predictions[frame_idx]
        
        pos = np.array([[a['x'], a['y']] for a in frame['atoms']])
        ord1 = np.array([a['ord1'] for a in frame['atoms']])
        ord2 = np.array([a['ord2'] for a in frame['atoms']])
        # Hexatic order magnitude |q6|
        hex_order = np.sqrt(ord1**2 + ord2**2)
        
        xlo, xhi, ylo, yhi = frame['box']
        
        # Left Panel: Hexatic Order
        sc1 = ax1.scatter(pos[:, 0], pos[:, 1], c=hex_order, cmap='viridis', s=10)
        ax1.set_title(f"Local Hexatic Order (|q6|) - Frame {frame_idx}")
        ax1.set_xlim(xlo, xhi)
        ax1.set_ylim(ylo, yhi)
        ax1.set_aspect('equal')
        
        # Right Panel: Propensity Heatmap using Triangulation for continuous field
        # We use matplotlib tricontourf
        triang = mtri.Triangulation(pos[:, 0], pos[:, 1])
        tcf = ax2.tricontourf(triang, preds, levels=20, cmap='hot')
        ax2.set_title(f"EGNN Predicted Propensity - Frame {frame_idx}")
        ax2.set_xlim(xlo, xhi)
        ax2.set_ylim(ylo, yhi)
        ax2.set_aspect('equal')
        
        return ax1, ax2

    print("Rendering MP4 animation...")
    ani = FuncAnimation(fig, update, frames=len(frames), blit=False)
    # Use imageio-ffmpeg (bundled binary, no system ffmpeg required)
    import imageio
    from matplotlib.animation import FFMpegWriter
    import matplotlib
    import imageio_ffmpeg
    matplotlib.rcParams['animation.ffmpeg_path'] = imageio_ffmpeg.get_ffmpeg_exe()
    writer = FFMpegWriter(fps=5)
    ani.save(output_mp4, writer=writer, dpi=150)
    print(f"Saved animation to {output_mp4}")

if __name__ == "__main__":
    create_softness_map_animation('trajectory_2k.dump', output_mp4='propensity_animation.mp4')
