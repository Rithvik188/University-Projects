import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.spatial import cKDTree
from scipy.stats import pearsonr, gaussian_kde
from torch_geometric.data import Data

from egnn_attentional import AttentionalEGNNModel
from prepare_80k_labels import parse_two_frames, SIGMA, R_SMOOTH, spatial_gaussian_smooth

DUMP_V1 = 'trajectory_80k.dump'
DUMP_V2 = 'trajectory_80k_v2.dump'
MODEL_PATH = 'egnn_universal_model.pth'
OUT_IMAGE = 'final_evaluation_dashboard.png'

def get_frame_data(dump_file, target_t0, f_act):
    pos0, pos50, box = parse_two_frames(dump_file, target_t0, target_t0 + 50000)
    if len(pos0) == 0:
        return None
        
    Lx = box[1] - box[0]
    Ly = box[3] - box[2]
    
    N = len(pos0)
    vec = torch.zeros((N, 2), dtype=torch.float)
    x_feat = torch.zeros((N, 4), dtype=torch.float)
    
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
                    x_feat[idx, 3] = f_act
                    i += 1
                break
            else:
                i += 1
        else:
            i += 1
            
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
    
    data = Data(x=x_feat, pos=pos, vec=vec, edge_index=edge_index)
    return data, gt_log

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    model = AttentionalEGNNModel(
        scalar_in_dim=4, vector_dim=2, hidden_dim=64, edge_dim=32, num_layers=4, heads=4
    ).to(device)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.eval()

    # Temporal Stability Data
    time_steps = list(range(60000, 470001, 50000))
    r_v1_history = []
    r_v2_history = []
    
    gt_real_all = []
    pred_real_all = []
    
    # Store V2 T=60k specifically for Plot 1
    v2_60k_gt_real = None
    v2_60k_pred_real = None
    
    print("Collecting Temporal Inference Data...")
    for t in time_steps:
        print(f"  Evaluating T={t}...")
        
        # V1
        res_v1 = get_frame_data(DUMP_V1, t, 1.0)
        if res_v1:
            data_v1, gt_log_v1 = res_v1
            data_v1 = data_v1.to(device)
            with torch.no_grad():
                pred_log_v1 = model(data_v1).cpu().numpy().flatten()
            r_v1, _ = pearsonr(pred_log_v1, gt_log_v1)
            r_v1_history.append((t, r_v1))
            
            if t == 60000:
                gt_real_all.append(np.exp(gt_log_v1))
                pred_real_all.append(np.exp(pred_log_v1))
                
        # V2
        res_v2 = get_frame_data(DUMP_V2, t, 1.25)
        if res_v2:
            data_v2, gt_log_v2 = res_v2
            data_v2 = data_v2.to(device)
            with torch.no_grad():
                pred_log_v2 = model(data_v2).cpu().numpy().flatten()
            r_v2, _ = pearsonr(pred_log_v2, gt_log_v2)
            r_v2_history.append((t, r_v2))
            
            if t == 60000:
                v2_60k_gt_real = np.exp(gt_log_v2)
                v2_60k_pred_real = np.exp(pred_log_v2)
                gt_real_all.append(v2_60k_gt_real)
                pred_real_all.append(v2_60k_pred_real)
                
    # Unpack temporal data
    t_v1, r_v1_vals = zip(*r_v1_history)
    t_v2, r_v2_vals = zip(*r_v2_history)
    
    # Combine GT/Pred for Plot 3 (Using T=60k combined V1 + V2)
    gt_real_combined = np.concatenate(gt_real_all)
    pred_real_combined = np.concatenate(pred_real_all)

    print("Generating Dashboard Plots...")
    sns.set_theme(style="whitegrid", context="paper", font_scale=1.5)
    fig, axes = plt.subplots(1, 3, figsize=(24, 7))

    # Plot 1: Heavy-Tail Prediction (PDF)
    ax = axes[0]
    sns.kdeplot(v2_60k_gt_real, log_scale=True, label="Ground Truth", color="black", linestyle="--", linewidth=2.5, ax=ax)
    sns.kdeplot(v2_60k_pred_real, log_scale=True, label="EGNN Prediction", color="darkorange", linewidth=2.5, ax=ax)
    ax.set_title("V2 (F_act=1.25) Displacement Distribution", fontweight='bold')
    ax.set_xlabel("Mobility (Squared Displacement)")
    ax.set_ylabel("Density")
    ax.legend()
    
    # Plot 2: Temporal Stability
    ax = axes[1]
    ax.plot(np.array(t_v1)/1000, r_v1_vals, marker='o', markersize=8, linewidth=2.5, label="V1 (F_act=1.00)", color="teal")
    ax.plot(np.array(t_v2)/1000, r_v2_vals, marker='s', markersize=8, linewidth=2.5, label="V2 (F_act=1.25)", color="crimson")
    ax.set_title("Generalization Temporal Stability", fontweight='bold')
    ax.set_xlabel("Simulation Time (x10³ Steps)")
    ax.set_ylabel("Pearson Correlation (R)")
    ax.set_ylim(0.4, 0.8)
    ax.legend(loc="lower right")

    # Plot 3: 1-to-1 Scatter
    ax = axes[2]
    # To avoid plotting 160k points which makes PDF/PNG huge, we sample 10,000 points
    idx = np.random.choice(len(gt_real_combined), 10000, replace=False)
    ax.scatter(gt_real_combined[idx], pred_real_combined[idx], alpha=0.15, color="indigo", edgecolor="none", s=10)
    
    # Perfect y=x line
    min_val = min(gt_real_combined.min(), pred_real_combined.min())
    max_val = max(gt_real_combined.max(), pred_real_combined.max())
    ax.plot([min_val, max_val], [min_val, max_val], 'k--', linewidth=2, label="Perfect Alignment (y=x)")
    
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_title("Predicted vs True Mobility (V1+V2)", fontweight='bold')
    ax.set_xlabel("Ground Truth Mobility")
    ax.set_ylabel("Predicted Mobility")
    ax.legend()

    plt.tight_layout()
    plt.savefig(OUT_IMAGE, dpi=300, bbox_inches='tight')
    print(f"Successfully generated {OUT_IMAGE}")

if __name__ == '__main__':
    main()
