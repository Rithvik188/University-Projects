"""
external_validation_prep.py
────────────────────────────────────────────────────────────────
A modular, teacher-ready script that:
  1. Accepts raw particle positions + box_size from any source.
  2. Computes Voronoi Area and q6 features from scratch.
  3. Loads our trained EGNN and produces a per-particle "Softness Map".

Works with ANY number of frames or particles — no hard-coded shapes.

Usage (standalone, pointing to a file):
    python external_validation_prep.py \
        --positions my_traj.npy \
        --box_size 34.7 \
        --model glass_gnn_model.pth \
        --out softness_map.npy

Usage (import as a library in a notebook):
    from external_validation_prep import prepare_external_data, inference_loop
    features = prepare_external_data(positions_array, box_size=L)
    softness = inference_loop(features, model_path='glass_gnn_model.pth')
────────────────────────────────────────────────────────────────
"""
from __future__ import annotations
import argparse
import numpy as np
import torch
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from scipy.spatial import Voronoi, cKDTree
from tqdm import tqdm

# ─── Feature Functions ────────────────────────────────────────────────────────

def _voronoi_area(points: np.ndarray, L: float) -> np.ndarray:
    """Periodic Voronoi cell areas via 3×3 tiling. Returns (N,) array."""
    n     = len(points)
    tiled = np.vstack([points + np.array([dx * L, dy * L])
                       for dx in [-1, 0, 1] for dy in [-1, 0, 1]])
    vor   = Voronoi(tiled)
    areas = np.zeros(n)
    for i in range(n):
        region = vor.regions[vor.point_region[4 * n + i]]
        if -1 in region or len(region) == 0:
            areas[i] = np.nan
            continue
        v = vor.vertices[region]
        x, y = v[:, 0], v[:, 1]
        areas[i] = 0.5 * abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))
    # Replace NaNs (boundary artefacts) with mean
    nanmask = np.isnan(areas)
    if nanmask.any():
        areas[nanmask] = np.nanmean(areas)
    return areas


def _q6(points: np.ndarray, L: float, rc: float = 1.5) -> np.ndarray:
    """2-D hexatic bond-order parameter. Returns (N,) array."""
    tree  = cKDTree(points, boxsize=[L, L])
    pairs = tree.query_pairs(r=rc, output_type='ndarray')
    q6c   = np.zeros(len(points), dtype=complex)
    nc    = np.zeros(len(points))
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


def _build_radius_graph(pts: np.ndarray, r_cut: float = 2.0) -> torch.Tensor:
    tree  = cKDTree(pts)
    pairs = tree.query_pairs(r=r_cut, output_type='ndarray')
    if len(pairs) > 0:
        eij = pairs.T
        ei  = torch.tensor(np.hstack([eij, eij[[1, 0]]]), dtype=torch.long)
    else:
        ei  = torch.empty((2, 0), dtype=torch.long)
    return ei


# ─── Public API ───────────────────────────────────────────────────────────────

def prepare_external_data(
    positions: np.ndarray,
    box_size: float,
    r_cut_graph: float = 2.0,
    r_cut_q6: float = 1.5,
    verbose: bool = True,
) -> list[Data]:
    """
    Build a list of PyG Data objects from raw positions.

    Parameters
    ----------
    positions   : np.ndarray, shape (F, N, 2) or (N, 2)
                  F frames, N particles, 2D coordinates.
    box_size    : float — length of the square periodic box.
    r_cut_graph : float — neighbour cutoff for radius graph (default 2.0).
    r_cut_q6    : float — neighbour cutoff for q6 (default 1.5).
    verbose     : bool  — show progress bars.

    Returns
    -------
    List of torch_geometric.data.Data objects, one per frame.
    """
    if positions.ndim == 2:           # (N, 2) → (1, N, 2)
        positions = positions[None]

    F, N, _ = positions.shape
    L        = box_size
    if verbose:
        print(f"  Frames={F}, Particles={N}, Box={L:.4f}")

    raw_feat = np.zeros((F, N, 2))
    iterator = tqdm(range(F), desc="Computing features") if verbose else range(F)
    for f in iterator:
        raw_feat[f, :, 0] = _voronoi_area(positions[f], L)
        raw_feat[f, :, 1] = _q6(positions[f], L, rc=r_cut_q6)

    # Global normalisation (consistent with training pipeline)
    mu  = raw_feat.mean(axis=(0, 1), keepdims=True)
    sig = raw_feat.std( axis=(0, 1), keepdims=True) + 1e-8
    feat_norm = (raw_feat - mu) / sig

    dataset  = []
    g_iter   = tqdm(range(F), desc="Building graphs") if verbose else range(F)
    for f in g_iter:
        pts  = positions[f]
        ei   = _build_radius_graph(pts, r_cut=r_cut_graph)
        data = Data(
            x          = torch.tensor(feat_norm[f], dtype=torch.float),
            pos        = torch.tensor(pts,          dtype=torch.float),
            edge_index = ei,
        )
        dataset.append(data)

    return dataset


def inference_loop(
    dataset: list[Data],
    model_path: str = 'glass_gnn_model.pth',
    batch_size: int = 16,
    device_str: str = 'auto',
    verbose: bool = True,
) -> np.ndarray:
    """
    Run EGNN inference and return a per-particle softness map.

    Parameters
    ----------
    dataset    : output of prepare_external_data()
    model_path : path to .pth weights file
    batch_size : DataLoader batch size
    device_str : 'auto' | 'cuda' | 'cpu'
    verbose    : bool

    Returns
    -------
    np.ndarray, shape (F*N,) — predicted propensity (softness) for every
    particle in every frame, in the same order as the input dataset.
    """
    from model import EGNNModel   # imported here so the file is standalone-importable

    if device_str == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(device_str)

    if verbose:
        print(f"  Inference device: {device}")

    model = EGNNModel(in_features=2, hidden_dim=64, num_layers=4).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model.eval()

    loader    = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    all_preds = []
    g_iter    = tqdm(loader, desc="Inference") if verbose else loader

    with torch.no_grad():
        for batch in g_iter:
            batch      = batch.to(device)
            preds      = model(batch)
            all_preds.extend(preds.cpu().numpy())

    softness = np.array(all_preds)
    if verbose:
        print(f"  Softness map shape: {softness.shape}")
        print(f"  Range: [{softness.min():.4f}, {softness.max():.4f}]")
    return softness


# ─── CLI entry point ──────────────────────────────────────────────────────────

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Prepare external data and run EGNN softness inference.'
    )
    parser.add_argument('--positions',  type=str, required=True,
                        help='Path to .npy positions array, shape (F,N,2) or (N,2)')
    parser.add_argument('--box_size',   type=float, required=True,
                        help='Periodic box side length (same units as positions)')
    parser.add_argument('--model',      type=str, default='glass_gnn_model.pth',
                        help='Path to trained EGNN weights (.pth)')
    parser.add_argument('--out',        type=str, default='softness_map.npy',
                        help='Output path for the softness map')
    parser.add_argument('--device',     type=str, default='auto',
                        choices=['auto', 'cuda', 'cpu'])
    parser.add_argument('--batch_size', type=int, default=16)
    args = parser.parse_args()

    print(f"\n{'='*55}")
    print("External Validation Pipeline — Active Glass EGNN")
    print(f"{'='*55}")
    print(f"Input  : {args.positions}")
    print(f"Box    : {args.box_size}")
    print(f"Model  : {args.model}")
    print(f"Output : {args.out}\n")

    positions = np.load(args.positions)
    print(f"Loaded positions: {positions.shape}\n")

    print("Step 1/2 — Preparing data…")
    dataset = prepare_external_data(positions, box_size=args.box_size)

    print("\nStep 2/2 — Running inference…")
    softness = inference_loop(dataset,
                              model_path=args.model,
                              batch_size=args.batch_size,
                              device_str=args.device)

    np.save(args.out, softness)
    print(f"\nSaved softness map → {args.out}  shape={softness.shape}")
    print("Done ✓")
