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
OUT_VIDEO = 'rolling_animation.mp4'

SIGMA = 2.0
R_SMOOTH = 4 * SIGMA

def extract_all_frames():
    """Extracts all frames into memory (since file is just 36MB, this is fast)."""
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

def quick_train_model(model, f_now, f_future, Lx, Ly, device):
    print("Quickly training on a subgraph of Frame 0 to get realistic weights...")
    delta = f_future['pos'] - f_now['pos']
    delta[:, 0] -= Lx * np.round(delta[:, 0] / Lx)
    delta[:, 1] -= Ly * np.round(delta[:, 1] / Ly)
    sq_disp = (delta ** 2).sum(axis=1)
    
    gt = spatial_gaussian_smooth(f_now['pos'], sq_disp, SIGMA, R_SMOOTH)
    gt = (gt - gt.min()) / (gt.max() - gt.min() + 1e-12)
    
    # Subgraph of first 5000 nodes for fast training
    N_sub = 5000
    pos_sub = f_now['pos'][:N_sub]
    vec_sub = f_now['vec'][:N_sub]
    feat_sub = f_now['feat'][:N_sub]
    gt_sub = gt[:N_sub]
    
    pos = torch.tensor(pos_sub, dtype=torch.float)
    vec = torch.tensor(vec_sub, dtype=torch.float)
    x_feat = torch.tensor(feat_sub, dtype=torch.float)
    x_feat = (x_feat - x_feat.mean(dim=0, keepdim=True)) / (x_feat.std(dim=0, keepdim=True) + 1e-8)
    y = torch.tensor(gt_sub, dtype=torch.float)
    
    tree = cKDTree(pos_sub)
    pairs = tree.query_pairs(r=4.0, output_type='ndarray')
    if len(pairs) > 0:
        edges_ij = pairs.T
        edges_ji = np.vstack([edges_ij[1], edges_ij[0]])
        edge_index = torch.tensor(np.hstack([edges_ij, edges_ji]), dtype=torch.long)
    else:
        edge_index = torch.empty((2, 0), dtype=torch.long)
        
    data = Data(x=x_feat, pos=pos, vec=vec, edge_index=edge_index, y=y).to(device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
    criterion = torch.nn.MSELoss()
    model.train()
    
    for epoch in range(15):
        optimizer.zero_grad()
        out = model(data)
        loss = criterion(out, data.y)
        loss.backward()
        optimizer.step()
        
    print("Quick training done.")
    model.eval()

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    frames, box = extract_all_frames()
    Lx = box[1] - box[0]
    Ly = box[3] - box[2]
    
    # Load model (or quick train if no weights)
    model = AttentionalEGNNModel(
        scalar_in_dim=3, vector_dim=2, hidden_dim=64, edge_dim=32, num_layers=4, heads=4
    ).to(device)
    
    if os.path.exists(MODEL_PATH):
        model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
        model.eval()
        print("Model loaded.")
    else:
        quick_train_model(model, frames[0], frames[5], Lx, Ly, device)
    
    # Pre-compute labels to find global vmin and vmax
    all_gt = []
    all_pred = []
    
    window = 5 # 5 frames = 50,000 steps
    
    plots = []
    
    print("Running rolling inference...")
    with imageio.get_writer(OUT_VIDEO, fps=2) as writer:
        for i in range(len(frames) - window):
            print(f"Processing frame index {i} -> {i+window} ...")
            f_now = frames[i]
            f_future = frames[i+window]
            
            # Ground truth
            delta = f_future['pos'] - f_now['pos']
            delta[:, 0] -= Lx * np.round(delta[:, 0] / Lx)
            delta[:, 1] -= Ly * np.round(delta[:, 1] / Ly)
            sq_disp = (delta ** 2).sum(axis=1)
            
            gt = spatial_gaussian_smooth(f_now['pos'], sq_disp, SIGMA, R_SMOOTH)
            gt = (gt - gt.min()) / (gt.max() - gt.min() + 1e-12)
            
            # Predict
            pos = torch.tensor(f_now['pos'], dtype=torch.float)
            vec = torch.tensor(f_now['vec'], dtype=torch.float)
            x_feat = torch.tensor(f_now['feat'], dtype=torch.float)
            
            # Standardize feat
            x_feat = (x_feat - x_feat.mean(dim=0, keepdim=True)) / (x_feat.std(dim=0, keepdim=True) + 1e-8)
            
            tree = cKDTree(pos.numpy())
            pairs = tree.query_pairs(r=4.0, output_type='ndarray')
            edges_ij = pairs.T
            edges_ji = np.vstack([edges_ij[1], edges_ij[0]])
            edge_index = torch.tensor(np.hstack([edges_ij, edges_ji]), dtype=torch.long)
            
            data = Data(x=x_feat, pos=pos, vec=vec, edge_index=edge_index)
            data = data.to(device)
            
            with torch.no_grad():
                pred = model(data).cpu().numpy().flatten()
            
            # Clip or normalize pred to match visual aesthetics (or use global scale)
            # We will use [0, 1] for both for simplicity and comparison
            pred = (pred - pred.min()) / (pred.max() - pred.min() + 1e-12)
            
            # Render plot
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
            
            sc1 = ax1.scatter(pos[:, 0], pos[:, 1], c=gt, cmap='inferno', s=1.0, vmin=0, vmax=1)
            ax1.set_title(f"Ground Truth (T={f_now['ts']} to {f_future['ts']})")
            ax1.set_xlim(box[0], box[1])
            ax1.set_ylim(box[2], box[3])
            ax1.set_aspect('equal')
            plt.colorbar(sc1, ax=ax1, fraction=0.046, pad=0.04)
            
            sc2 = ax2.scatter(pos[:, 0], pos[:, 1], c=pred, cmap='inferno', s=1.0, vmin=0, vmax=1)
            ax2.set_title(f"EGNN Prediction (T={f_now['ts']})")
            ax2.set_xlim(box[0], box[1])
            ax2.set_ylim(box[2], box[3])
            ax2.set_aspect('equal')
            plt.colorbar(sc2, ax=ax2, fraction=0.046, pad=0.04)
            
            plt.tight_layout()
            
            # Save frame to memory buffer
            buf = io.BytesIO()
            fig.savefig(buf, format='png', dpi=100, bbox_inches='tight')
            buf.seek(0)
            image = imageio.imread(buf)
            writer.append_data(image)
            
            plt.close(fig)
            del data
            gc.collect()

    print(f"Saved {OUT_VIDEO}")

if __name__ == "__main__":
    main()
