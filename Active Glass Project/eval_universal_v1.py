import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import cKDTree
from scipy.stats import pearsonr
from torch_geometric.data import Data

from egnn_attentional import AttentionalEGNNModel
from prepare_80k_labels import parse_two_frames, TARGET_T0, TARGET_T50, SIGMA, R_SMOOTH, spatial_gaussian_smooth

DUMP_FILE = 'trajectory_80k.dump'
MODEL_PATH = 'egnn_universal_model.pth'
OUT_IMAGE = 'static_universal_v1.png'
F_ACT = 1.0

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    print(f"Loading graph for V1 T=60000 from {DUMP_FILE}...")
    pos0, pos50, box = parse_two_frames(DUMP_FILE, TARGET_T0, TARGET_T50)
    Lx = box[1] - box[0]
    Ly = box[3] - box[2]
    
    # Reconstruct features for T=60000
    N = len(pos0)
    print(f"Nodes extracted: {N}")
    vec = torch.zeros((N, 2), dtype=torch.float)
    x_feat = torch.zeros((N, 4), dtype=torch.float) # 4 features
    
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
                    x_feat[idx, 3] = F_ACT # Append F_act
                    i += 1
                break
            else:
                i += 1
        else:
            i += 1
            
    # Standardize first 3 features
    for col in range(3):
        x_feat[:, col] = (x_feat[:, col] - x_feat[:, col].mean()) / (x_feat[:, col].std() + 1e-8)
    
    # Ground Truth computation
    delta = pos50 - pos0
    delta[:, 0] -= Lx * np.round(delta[:, 0] / Lx)
    delta[:, 1] -= Ly * np.round(delta[:, 1] / Ly)
    sq_disp = (delta ** 2).sum(axis=1)
    
    gt_smoothed = spatial_gaussian_smooth(pos0, sq_disp, SIGMA, R_SMOOTH)
    gt_log = np.log(gt_smoothed + 1e-8)
    
    # Prepare Graph
    pos = torch.tensor(pos0, dtype=torch.float)
    tree = cKDTree(pos.numpy())
    pairs = tree.query_pairs(r=4.0, output_type='ndarray')
    edges_ij = pairs.T
    edges_ji = np.vstack([edges_ij[1], edges_ij[0]])
    edge_index = torch.tensor(np.hstack([edges_ij, edges_ji]), dtype=torch.long)
    
    data = Data(x=x_feat, pos=pos, vec=vec, edge_index=edge_index)
    data = data.to(device)
    
    # Load Universal Model (scalar_in_dim=4)
    model = AttentionalEGNNModel(
        scalar_in_dim=4, vector_dim=2, hidden_dim=64, edge_dim=32, num_layers=4, heads=4
    ).to(device)
    
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.eval()
    
    print("Running Universal Model full-batch inference on V1 T=60000...")
    with torch.no_grad():
        pred_log = model(data).cpu().numpy().flatten()
        
    r_val, _ = pearsonr(pred_log, gt_log)
    print(f"Universal Model V1 Pearson R: {r_val:.4f}")
        
    # Inverse Transformation for Rendering
    print("Applying np.exp() inverse transformation for rendering...")
    gt_real = np.exp(gt_log)
    pred_real = np.exp(pred_log)
    
    # Find shared color scale based on GT
    vmin = np.percentile(gt_real, 5)
    vmax = np.percentile(gt_real, 95)
    
    print(f"Color scale: vmin={vmin:.2f}, vmax={vmax:.2f}")
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
    
    sc1 = ax1.scatter(pos0[:, 0], pos0[:, 1], c=gt_real, cmap='plasma', s=1.0, vmin=vmin, vmax=vmax)
    ax1.set_title(f"V1 Ground Truth Propensity (T=60k to 110k)")
    ax1.set_xlim(box[0], box[1])
    ax1.set_ylim(box[2], box[3])
    ax1.set_aspect('equal')
    plt.colorbar(sc1, ax=ax1, fraction=0.046, pad=0.04)
    
    sc2 = ax2.scatter(pos0[:, 0], pos0[:, 1], c=pred_real, cmap='plasma', s=1.0, vmin=vmin, vmax=vmax)
    ax2.set_title(f"Universal EGNN Prediction (V1 T=60k)\nPearson R = {r_val:.4f}")
    ax2.set_xlim(box[0], box[1])
    ax2.set_ylim(box[2], box[3])
    ax2.set_aspect('equal')
    plt.colorbar(sc2, ax=ax2, fraction=0.046, pad=0.04)
    
    plt.tight_layout()
    plt.savefig(OUT_IMAGE, format='png', dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved {OUT_IMAGE}")

if __name__ == '__main__':
    main()
