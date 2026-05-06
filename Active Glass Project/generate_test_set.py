"""
generate_test_set.py
Runs a fresh ABP simulation with seed=42 (independent of the training set),
saves test_trajectory.npy, test_labels.npy, then computes Voronoi + q6 features
and packages everything into blind_test_data.pt for evaluation.
"""
import numpy as np
import torch
from torch_geometric.data import Data
from scipy.spatial import Voronoi, cKDTree
from tqdm import tqdm

# ── Reproducible, independent seed ────────────────────────────────────────────
SEED = 42
np.random.seed(SEED)

# ── Simulation parameters (identical physics, new seed) ───────────────────────
N         = 1024
phi       = 0.85
v0        = 1.0
Dr        = 1.0
dt        = 0.001
eq_steps  = 100_000
prod_steps = 20_000
save_freq = 100

sigma   = 1.0
epsilon = 1.0

particle_area = np.pi * (sigma / 2) ** 2
total_area    = N * particle_area / phi
L             = np.sqrt(total_area)
print(f"Box size L = {L:.4f}, phi = {phi}, N = {N}, seed = {SEED}")

# ── Initial conditions ─────────────────────────────────────────────────────────
grid_size = int(np.ceil(np.sqrt(N)))
spacing   = L / grid_size
gx, gy    = np.meshgrid(np.arange(grid_size) * spacing,
                         np.arange(grid_size) * spacing)
pos       = np.vstack((gx.flatten(), gy.flatten())).T[:N]
pos      += np.random.uniform(-0.1 * spacing, 0.1 * spacing, size=pos.shape)
pos       = np.mod(pos, L)
theta     = np.random.uniform(0, 2 * np.pi, N)

# ── Force kernel ───────────────────────────────────────────────────────────────
def compute_forces(pos, L, sigma, epsilon):
    tree   = cKDTree(pos, boxsize=[L, L])
    rc     = (2 ** (1 / 6)) * sigma
    pairs  = tree.query_pairs(r=rc, output_type='ndarray')
    F      = np.zeros_like(pos)
    if len(pairs) > 0:
        i, j   = pairs[:, 0], pairs[:, 1]
        diff   = pos[i] - pos[j]
        diff  -= L * np.round(diff / L)
        r2     = np.maximum(np.sum(diff ** 2, axis=1), 1e-12)
        r6     = (sigma ** 2 / r2) ** 3
        fmag   = 48 * epsilon * (r6 ** 2 - 0.5 * r6) / r2
        fp     = diff * fmag[:, None]
        cap    = np.linalg.norm(fp, axis=1)
        mask   = cap > 100.0
        fp[mask] *= (100.0 / cap[mask])[:, None]
        np.add.at(F, i,  fp)
        np.add.at(F, j, -fp)
    return F

# ── Step helper ────────────────────────────────────────────────────────────────
def abp_step(pos, theta, L):
    F          = compute_forces(pos, L, sigma, epsilon)
    dx         = (F[:, 0] + v0 * np.cos(theta)) * dt
    dy         = (F[:, 1] + v0 * np.sin(theta)) * dt
    pos[:, 0] += dx
    pos[:, 1] += dy
    theta     += np.sqrt(2 * Dr * dt) * np.random.randn(N)
    return pos, theta, dx, dy

# ── Equilibration ───────────────────────────────────────────────────────────────
unwrapped = pos.copy()
print("Equilibrating…")
for _ in tqdm(range(eq_steps), desc="Equilibration"):
    pos, theta, dx, dy = abp_step(pos, theta, L)
    unwrapped[:, 0] += dx
    unwrapped[:, 1] += dy
    pos = np.mod(pos, L)

# ── Production ─────────────────────────────────────────────────────────────────
saved_frames = prod_steps // save_freq
trajectory   = np.zeros((saved_frames, N, 2))
frame_idx    = 0
prod_start   = unwrapped.copy()

print("Production…")
for step in tqdm(range(prod_steps), desc="Production"):
    pos, theta, dx, dy = abp_step(pos, theta, L)
    unwrapped[:, 0] += dx
    unwrapped[:, 1] += dy
    pos = np.mod(pos, L)
    if step % save_freq == 0 and frame_idx < saved_frames:
        trajectory[frame_idx] = pos.copy()
        frame_idx += 1

# Propensity = total displacement magnitude over production run
test_labels = np.linalg.norm(unwrapped - prod_start, axis=1)   # (N,)

np.save('test_trajectory.npy', trajectory)
np.save('test_labels.npy',     test_labels)
print(f"Saved test_trajectory.npy {trajectory.shape}, test_labels.npy {test_labels.shape}")

# ── Feature extraction (Voronoi area + q6) ────────────────────────────────────

def get_voronoi_area(points, L):
    """Periodic Voronoi areas via 3×3 tiling."""
    tiled = np.vstack([points + np.array([dx * L, dy * L])
                       for dx in [-1, 0, 1] for dy in [-1, 0, 1]])
    vor   = Voronoi(tiled)
    n     = len(points)
    areas = np.zeros(n)
    for i in range(n):
        idx     = 4 * n + i          # centre-tile offset
        region  = vor.regions[vor.point_region[idx]]
        if -1 in region or len(region) == 0:
            areas[i] = np.nan
            continue
        verts   = vor.vertices[region]
        x, y    = verts[:, 0], verts[:, 1]
        areas[i] = 0.5 * abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))
    return areas

def get_q6(points, L, rc=1.5):
    """2-D bond-order parameter q6."""
    tree   = cKDTree(points, boxsize=[L, L])
    pairs  = tree.query_pairs(r=rc, output_type='ndarray')
    q6c    = np.zeros(len(points), dtype=complex)
    nc     = np.zeros(len(points))
    if len(pairs) > 0:
        i, j   = pairs[:, 0], pairs[:, 1]
        diff   = points[j] - points[i]
        diff  -= L * np.round(diff / L)
        ang    = np.arctan2(diff[:, 1], diff[:, 0])
        np.add.at(q6c, i, np.exp(6j * ang))
        np.add.at(nc,  i, 1)
        np.add.at(q6c, j, np.exp(6j * (ang + np.pi)))
        np.add.at(nc,  j, 1)
    nc[nc == 0] = 1
    return np.abs(q6c / nc)

print("Computing features (Voronoi + q6) for all frames…")
n_frames        = trajectory.shape[0]
test_features   = np.zeros((n_frames, N, 2))
for f in tqdm(range(n_frames), desc="Features"):
    pts                   = trajectory[f]
    test_features[f, :, 0] = get_voronoi_area(pts, L)
    test_features[f, :, 1] = get_q6(pts, L)

# Normalise using each frame's own statistics (same as training pipeline)
mean_f = test_features.mean(axis=(0, 1), keepdims=True)
std_f  = test_features.std(axis=(0, 1),  keepdims=True)
test_features_norm = (test_features - mean_f) / (std_f + 1e-8)

# ── Build PyG dataset ──────────────────────────────────────────────────────────
y_tensor = torch.tensor(test_labels, dtype=torch.float)   # (N,) per-particle
dataset  = []
print("Building radius graphs (r_cut=2.0)…")
for f in tqdm(range(n_frames), desc="Graphs"):
    pts_f   = trajectory[f]
    tree    = cKDTree(pts_f)
    pairs   = tree.query_pairs(r=2.0, output_type='ndarray')
    if len(pairs) > 0:
        eij     = pairs.T
        eji     = np.vstack([eij[1], eij[0]])
        ei      = torch.tensor(np.hstack([eij, eji]), dtype=torch.long)
    else:
        ei = torch.empty((2, 0), dtype=torch.long)
    data = Data(
        x         = torch.tensor(test_features_norm[f], dtype=torch.float),
        pos       = torch.tensor(pts_f,                 dtype=torch.float),
        edge_index= ei,
        y         = y_tensor,
    )
    dataset.append(data)

torch.save(dataset, 'blind_test_data.pt')
print(f"\nSaved blind_test_data.pt  ({len(dataset)} graphs, each with {N} nodes)")
print("Done — ready for evaluate_transferability.py")
