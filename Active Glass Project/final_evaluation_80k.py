import os
import gc
import numpy as np
import torch
import io
import matplotlib.pyplot as plt
from scipy.spatial import cKDTree
import imageio.v2 as imageio

from egnn_attentional import AttentionalEGNNModel
from torch_geometric.data import Data

DUMP_FILE = 'trajectory_80k.dump'
MODEL_PATH = 'egnn_attentional_model_80k.pth'
OUT_VIDEO = 'final_presentation_80k.mp4'

SIGMA = 2.0
R_SMOOTH = 4 * SIGMA
WINDOW = 5  # 5 frames = 50,000 steps

def extract_all_frames():
    """Extracts all frames into memory."""
    frames = []
    with open(DUMP_FILE, 'r') as fh:
        lines = fh.readlines()
        
    i = 0
    box = None
    while i < len(lines):
        tok = lines[i].strip()
        if tok == "ITEM: TIMESTEP":
            ts = int(lines[i + 1].strip())
            i += 2
        elif tok == "ITEM: NUMBER OF ATOMS":
            num_atoms = int(lines[i + 1].strip())
            i += 2
        elif tok.startswith("ITEM: BOX BOUNDS"):
            xlo, xhi = map(float, lines[i + 1].split()[:2])
            ylo, yhi = map(float, lines[i + 2].split()[:2])
            if box is None:
                box = (xlo, xhi, ylo, yhi)
            i += 4
        elif tok.startswith("ITEM: ATOMS"):
            i += 1
            ids = np.empty(num_atoms, dtype=np.int32)
            pos = np.empty((num_atoms, 2), dtype=np.float64)
            vec = np.empty((num_atoms, 2), dtype=np.float32)
            feat = np.empty((num_atoms, 3), dtype=np.float32)
            
            for k in range(num_atoms):
                parts = lines[i].split()
                ids[k] = int(parts[0])
                pos[k, 0] = float(parts[2])
                pos[k, 1] = float(parts[3])
                vec[k, 0] = float(parts[4])
                vec[k, 1] = float(parts[5])
                feat[k, 0] = float(parts[6])
                feat[k, 1] = float(parts[7])
                feat[k, 2] = float(parts[8])
                i += 1
                
            # Sort by id
            order = np.argsort(ids)
            frames.append({
                'ts': ts,
                'pos': pos[order],
                'vec': vec[order],
                'feat': feat[order]
            })
            print(f"Loaded frame {len(frames)} (ts={ts})")
        else:
            i += 1
            
    return frames, box

def spatial_gaussian_smooth(pos, values, sigma, r_cut):
    tree = cKDTree(pos)
    D = tree.sparse_distance_matrix(tree, max_distance=r_cut, output_type="coo_matrix")
    W = D.copy()
    W.data = np.exp(-0.5 * (W.data / sigma) ** 2)
    W_csr = W.tocsr()
    row_sum = np.asarray(W_csr.sum(axis=1)).ravel()
    smoothed = W_csr.dot(values) / row_sum
    return smoothed.astype(np.float32)

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    frames, box = extract_all_frames()
    Lx = box[1] - box[0]
    Ly = box[3] - box[2]
    
    print("Pre-computing global color scale across ALL frames...")
    global_gt_vals = []
    
    # We will also save the processed graph inputs and GT to avoid recomputing
    processed_data = []
    
    for i in range(len(frames) - WINDOW):
        f_now = frames[i]
        f_future = frames[i + WINDOW]
        
        # Ground truth
        delta = f_future['pos'] - f_now['pos']
        delta[:, 0] -= Lx * np.round(delta[:, 0] / Lx)
        delta[:, 1] -= Ly * np.round(delta[:, 1] / Ly)
        sq_disp = (delta ** 2).sum(axis=1)
        
        gt_smoothed = spatial_gaussian_smooth(f_now['pos'], sq_disp, SIGMA, R_SMOOTH)
        gt_log = np.log(gt_smoothed + 1e-8)
        
        # We need the inverse exponentiated GT to find global bounds
        gt_real = np.exp(gt_log)
        # Sample to avoid huge memory for global percentiles
        global_gt_vals.append(np.percentile(gt_real, [5, 95]))
        
        # Standardize features exactly as done during training
        pos = torch.tensor(f_now['pos'], dtype=torch.float)
        vec = torch.tensor(f_now['vec'], dtype=torch.float)
        x_feat = torch.tensor(f_now['feat'], dtype=torch.float)
        x_feat = (x_feat - x_feat.mean(dim=0, keepdim=True)) / (x_feat.std(dim=0, keepdim=True) + 1e-8)
        
        # Precompute edge_index to save time during rendering loop
        tree = cKDTree(pos.numpy())
        pairs = tree.query_pairs(r=4.0, output_type='ndarray')
        edges_ij = pairs.T
        edges_ji = np.vstack([edges_ij[1], edges_ij[0]])
        edge_index = torch.tensor(np.hstack([edges_ij, edges_ji]), dtype=torch.long)
        
        processed_data.append({
            'ts_start': f_now['ts'],
            'ts_end': f_future['ts'],
            'pos': f_now['pos'],
            'gt_real': gt_real,
            'data_obj': Data(x=x_feat, pos=pos, vec=vec, edge_index=edge_index)
        })
        print(f"Preprocessed frame index {i}")

    global_gt_vals = np.array(global_gt_vals)
    # Use the median of the 5th percentiles and 95th percentiles across time 
    # to avoid a single anomalous frame blowing out the scale
    vmin = np.median(global_gt_vals[:, 0])
    vmax = np.median(global_gt_vals[:, 1])
    print(f"Global Color Scale Locked: vmin={vmin:.2f}, vmax={vmax:.2f}")

    # Load Exact Model (No Retraining)
    model = AttentionalEGNNModel(
        scalar_in_dim=3, vector_dim=2, hidden_dim=64, edge_dim=32, num_layers=4, heads=4
    ).to(device)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.eval()
    print("Model loaded. Executing Temporal Generalization Phase...")
    
    with imageio.get_writer(OUT_VIDEO, fps=4) as writer:
        for i, item in enumerate(processed_data):
            print(f"Rendering frame {i+1}/{len(processed_data)} ...")
            data = item['data_obj'].to(device)
            gt_real = item['gt_real']
            
            with torch.no_grad():
                pred_log = model(data).cpu().numpy().flatten()
            
            pred_real = np.exp(pred_log)
            
            # Render plot
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
            
            sc1 = ax1.scatter(item['pos'][:, 0], item['pos'][:, 1], c=gt_real, cmap='plasma', s=1.0, vmin=vmin, vmax=vmax)
            ax1.set_title(f"Ground Truth (T={item['ts_start']} to {item['ts_end']})")
            ax1.set_xlim(box[0], box[1])
            ax1.set_ylim(box[2], box[3])
            ax1.set_aspect('equal')
            plt.colorbar(sc1, ax=ax1, fraction=0.046, pad=0.04)
            
            sc2 = ax2.scatter(item['pos'][:, 0], item['pos'][:, 1], c=pred_real, cmap='plasma', s=1.0, vmin=vmin, vmax=vmax)
            ax2.set_title(f"EGNN Prediction (T={item['ts_start']})")
            ax2.set_xlim(box[0], box[1])
            ax2.set_ylim(box[2], box[3])
            ax2.set_aspect('equal')
            plt.colorbar(sc2, ax=ax2, fraction=0.046, pad=0.04)
            
            plt.tight_layout()
            
            buf = io.BytesIO()
            fig.savefig(buf, format='png', dpi=100, bbox_inches='tight')
            buf.seek(0)
            image = imageio.imread(buf)
            writer.append_data(image)
            
            plt.close(fig)
            del data
            gc.collect()

    print(f"Masterpiece rendered: {OUT_VIDEO}")

if __name__ == "__main__":
    main()
