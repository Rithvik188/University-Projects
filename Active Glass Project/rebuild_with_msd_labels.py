"""
rebuild_with_msd_labels.py
────────────────────────────────────────────────────────────────
Rebuilds both the training and blind-test datasets with physically
meaningful per-frame MSD labels.

Strategy:
  For training data (seed original):
    - For each frame t, propensity_i(t) = ||x_i(t + dt_window) - x_i(t)||
    - We use a look-ahead window of `window` frames.
    - Frames within `window` of the end are dropped.
  
  For blind test data (seed=42):
    - Same procedure applied to test_trajectory.npy

This ensures:
  1. Each graph has its OWN per-particle propensity label (not shared).
  2. The label distribution is consistent between train/test sets.
  3. Transferability is physically meaningful.
────────────────────────────────────────────────────────────────
"""
import numpy as np
import torch
from torch_geometric.data import Data
from scipy.spatial import cKDTree
from tqdm import tqdm

WINDOW    = 10   # frames ahead for MSD
R_CUT     = 2.0  # radius graph cutoff

def get_features_normalized(trajectory):
    """Load and normalize Voronoi area + q6 features."""
    from scipy.spatial import Voronoi

    n_frames, N, _ = trajectory.shape
    L = np.sqrt(N * (np.pi * 0.25) / 0.85)

    def voronoi_area(pts):
        tiled = np.vstack([pts + np.array([dx*L, dy*L])
                           for dx in [-1,0,1] for dy in [-1,0,1]])
        vor = Voronoi(tiled)
        areas = np.zeros(N)
        for i in range(N):
            reg = vor.regions[vor.point_region[4*N+i]]
            if -1 in reg or len(reg)==0:
                areas[i] = np.nan; continue
            v = vor.vertices[reg]
            x,y = v[:,0], v[:,1]
            areas[i] = 0.5*abs(np.dot(x,np.roll(y,1))-np.dot(y,np.roll(x,1)))
        areas[np.isnan(areas)] = np.nanmean(areas)
        return areas

    def q6(pts):
        tree = cKDTree(pts, boxsize=[L,L])
        pairs = tree.query_pairs(r=1.5, output_type='ndarray')
        q6c = np.zeros(N, dtype=complex); nc = np.zeros(N)
        if len(pairs)>0:
            i,j = pairs[:,0], pairs[:,1]
            diff = pts[j]-pts[i]; diff -= L*np.round(diff/L)
            ang = np.arctan2(diff[:,1],diff[:,0])
            np.add.at(q6c, i, np.exp(6j*ang)); np.add.at(nc, i, 1)
            np.add.at(q6c, j, np.exp(6j*(ang+np.pi))); np.add.at(nc, j, 1)
        nc[nc==0]=1
        return np.abs(q6c/nc)

    feats = np.zeros((n_frames, N, 2))
    for f in tqdm(range(n_frames), desc="  Features", leave=False):
        feats[f,:,0] = voronoi_area(trajectory[f])
        feats[f,:,1] = q6(trajectory[f])

    mu  = feats.mean(axis=(0,1), keepdims=True)
    sig = feats.std(axis=(0,1),  keepdims=True) + 1e-8
    return (feats - mu) / sig, L

def build_dataset(trajectory, features_norm, labels_per_frame):
    """Build a list of PyG Data objects with per-frame per-node labels."""
    n_frames = len(labels_per_frame)
    L = np.sqrt(trajectory.shape[1] * (np.pi * 0.25) / 0.85)
    dataset = []
    for f in tqdm(range(n_frames), desc="  Graphs  ", leave=False):
        pts = trajectory[f]
        tree = cKDTree(pts)
        pairs = tree.query_pairs(r=R_CUT, output_type='ndarray')
        if len(pairs)>0:
            eij = pairs.T
            ei  = torch.tensor(np.hstack([eij, eij[[1,0]]]), dtype=torch.long)
        else:
            ei = torch.empty((2,0), dtype=torch.long)
        data = Data(
            x          = torch.tensor(features_norm[f], dtype=torch.float),
            pos        = torch.tensor(pts,              dtype=torch.float),
            edge_index = ei,
            y          = torch.tensor(labels_per_frame[f], dtype=torch.float),
        )
        dataset.append(data)
    return dataset

# ─── Training set ─────────────────────────────────────────────────────────────
print("=== Rebuilding TRAINING dataset with MSD labels ===")
traj_train = np.load('trajectory.npy')           # (200, 1024, 2)
n, N, _ = traj_train.shape
n_valid = n - WINDOW                              # frames with a valid look-ahead

# Compute per-frame MSD propensity
print(f"  Computing MSD with window={WINDOW} frames...")
msd_train = np.zeros((n_valid, N))
for f in range(n_valid):
    diff = traj_train[f+WINDOW] - traj_train[f]  # (N, 2)
    msd_train[f] = np.linalg.norm(diff, axis=1)  # (N,)

feats_train, L_train = get_features_normalized(traj_train[:n_valid])
dataset_train = build_dataset(traj_train[:n_valid], feats_train, msd_train)
torch.save(dataset_train, 'glass_graph_data.pt')
print(f"  Saved glass_graph_data.pt: {len(dataset_train)} graphs")
print(f"  Label range: [{msd_train.min():.4f}, {msd_train.max():.4f}]")

# ─── Blind test set ───────────────────────────────────────────────────────────
print("\n=== Rebuilding BLIND TEST dataset with MSD labels ===")
traj_test = np.load('test_trajectory.npy')       # (200, 1024, 2)
n, _, _ = traj_test.shape
n_valid_test = n - WINDOW

print(f"  Computing MSD with window={WINDOW} frames...")
msd_test = np.zeros((n_valid_test, N))
for f in range(n_valid_test):
    diff = traj_test[f+WINDOW] - traj_test[f]
    msd_test[f] = np.linalg.norm(diff, axis=1)

feats_test, _ = get_features_normalized(traj_test[:n_valid_test])
dataset_test  = build_dataset(traj_test[:n_valid_test], feats_test, msd_test)
torch.save(dataset_test, 'blind_test_data.pt')
print(f"  Saved blind_test_data.pt: {len(dataset_test)} graphs")
print(f"  Label range: [{msd_test.min():.4f}, {msd_test.max():.4f}]")

print("\nDone! Now re-run train_model.py, then evaluate_transferability.py.")
