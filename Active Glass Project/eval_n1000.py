import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import freud
from torch_geometric.data import Data

from egnn_attentional import AttentionalEGNNModel
from prepare_80k_labels import parse_two_frames  # We can still use this if we modify it for 1 frame

DUMP_FILE = 'mips_2025.1K.dump.2.0'
MODEL_PATH = 'egnn_universal_model.pth'
OUT_IMAGE = 'flock_heatmap_N1000.png'
F_ACT = 1.0  # Default assumption for F_act
SIGMA = 2.0
R_CUT = 2.0 * SIGMA  # 4.0

def load_single_frame(filepath):
    """Custom parser for a single LAMMPS dump frame since parse_two_frames expects 2."""
    with open(filepath, 'r') as fh:
        lines = fh.readlines()
        
    i = 0
    while i < len(lines):
        if lines[i].strip() == "ITEM: TIMESTEP":
            ts = int(lines[i+1].strip())
            i += 2
        elif lines[i].strip() == "ITEM: NUMBER OF ATOMS":
            num_atoms = int(lines[i+1].strip())
            i += 2
        elif lines[i].startswith("ITEM: BOX BOUNDS"):
            xlo, xhi = map(float, lines[i+1].split()[:2])
            ylo, yhi = map(float, lines[i+2].split()[:2])
            box = (xlo, xhi, ylo, yhi)
            i += 4
        elif lines[i].startswith("ITEM: ATOMS"):
            i += 1
            ids = np.empty(num_atoms, dtype=np.int32)
            pos = np.empty((num_atoms, 2), dtype=np.float64)
            vec = np.empty((num_atoms, 2), dtype=np.float64)
            for k in range(num_atoms):
                parts = lines[i].split()
                ids[k] = int(parts[0])
                pos[k, 0] = float(parts[2])
                pos[k, 1] = float(parts[3])
                # We need velocity/orientation for `vec`. Let's assume vx, vy are at 8, 9 
                # (from mips dump: x y xs ys xu yu vx vy)
                vec[k, 0] = float(parts[8])
                vec[k, 1] = float(parts[9])
                i += 1
                
            order = np.argsort(ids)
            return pos[order], vec[order], box
        else:
            i += 1

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    print(f"Loading graph from {DUMP_FILE}...")
    pos, vec, box_bounds = load_single_frame(DUMP_FILE)
    xlo, xhi, ylo, yhi = box_bounds
    Lx = xhi - xlo
    Ly = yhi - ylo
    
    N = len(pos)
    print(f"Nodes extracted: {N}, Box: {Lx}x{Ly}")
    
    # Use freud to compute features with PBC
    freud_box = freud.Box.from_box([Lx, Ly, 0])
    
    # 1. Voronoi Area
    vor = freud.locality.Voronoi()
    # Add z=0 to pos for freud
    pos_3d = np.zeros((N, 3))
    pos_3d[:, :2] = pos
    vor.compute((freud_box, pos_3d))
    A_Vor = vor.volumes
    
    # 2. Hexatic Order Parameter (q_6)
    hex_order = freud.order.Hexatic(k=6)
    hex_order.compute((freud_box, pos_3d))
    q_6_complex = hex_order.particle_order
    q_6_real = q_6_complex.real
    q_6_imag = q_6_complex.imag
    
    # Construct node features [A_Vor, q_6_real, q_6_imag, F_act]
    x_feat = np.zeros((N, 4), dtype=np.float32)
    x_feat[:, 0] = A_Vor
    x_feat[:, 1] = q_6_real
    x_feat[:, 2] = q_6_imag
    x_feat[:, 3] = F_ACT
    
    # Z-score standardisation (only on first 3 features)
    for col in range(3):
        x_feat[:, col] = (x_feat[:, col] - x_feat[:, col].mean()) / (x_feat[:, col].std() + 1e-8)
        
    # PBC Edges using freud
    aq = freud.locality.AABBQuery(freud_box, pos_3d)
    nlist = aq.query(pos_3d, dict(r_max=R_CUT, exclude_ii=True)).toNeighborList()
    edge_index = np.vstack((nlist.query_point_indices, nlist.point_indices)).astype(np.int64)
    
    # Create PyG Data
    data = Data(
        x=torch.tensor(x_feat, dtype=torch.float),
        pos=torch.tensor(pos, dtype=torch.float),
        vec=torch.tensor(vec, dtype=torch.float),
        edge_index=torch.tensor(edge_index, dtype=torch.long)
    ).to(device)
    
    print("Loading Universal Model...")
    model = AttentionalEGNNModel(
        scalar_in_dim=4, vector_dim=2, hidden_dim=64, edge_dim=32, num_layers=4, heads=4
    ).to(device)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.eval()
    
    print("Running Zero-Shot Inference...")
    with torch.no_grad():
        pred_log = model(data).cpu().numpy().flatten()
        
    # Inverse transformation
    pred_real = np.exp(pred_log)
    
    print("Generating Heatmap...")
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Color scaling
    vmin = np.percentile(pred_real, 5)
    vmax = np.percentile(pred_real, 95)
    
    sc = ax.scatter(pos[:, 0], pos[:, 1], c=pred_real, cmap='plasma', s=30.0, vmin=vmin, vmax=vmax)
    ax.set_title('Zero-Shot Active Flocking Prediction (N=1000)', fontsize=14)
    ax.set_xlabel(r'$x/\sigma$', fontsize=12)
    ax.set_ylabel(r'$y/\sigma$', fontsize=12)
    ax.set_xlim(xlo, xhi)
    ax.set_ylim(ylo, yhi)
    ax.set_aspect('equal')
    
    cbar = plt.colorbar(sc)
    cbar.set_label(r'$\log(\mathcal{P}_i)$ over $\Delta t = 50,000$ steps', rotation=270, labelpad=15, fontsize=12)
    
    plt.tight_layout()
    plt.savefig(OUT_IMAGE, format='png', dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved {OUT_IMAGE}")

if __name__ == '__main__':
    main()
