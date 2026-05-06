"""
Phase 1 – Ground Truth Extraction
  T=0   → timestep 60000  (frame index 0)
  T=50k → timestep 110000 (frame index 5, step interval = 10000)

Outputs:
  labels_80k.npy  –  Gaussian-smoothed squared displacement, shape (80000,)
"""

import numpy as np
from scipy.spatial import cKDTree
from scipy import sparse

DUMP_FILE  = "trajectory_80k.dump"
OUT_FILE   = "labels_80k.npy"
TARGET_T0  = 60000
TARGET_T50 = 110000
SIGMA      = 2.0   # Gaussian smoothing length (simulation units)
R_SMOOTH   = 4 * SIGMA


# ── robust LAMMPS dump parser ──────────────────────────────────────────────
def parse_two_frames(filepath, t0, t50):
    """Return (atoms_t0, atoms_t50, box) for the two requested timesteps."""
    results = {}
    box = None

    with open(filepath, "r") as fh:
        lines = fh.readlines()

    i = 0
    n = len(lines)
    while i < n:
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
            box = (xlo, xhi, ylo, yhi)
            i += 4          # skip z-bounds line

        elif tok.startswith("ITEM: ATOMS"):
            i += 1          # skip header line
            if ts not in (t0, t50):
                i += num_atoms
                continue

            ids  = np.empty(num_atoms, dtype=np.int32)
            pos  = np.empty((num_atoms, 2), dtype=np.float64)

            for k in range(num_atoms):
                parts = lines[i].split()
                ids[k]    = int(parts[0])
                pos[k, 0] = float(parts[2])
                pos[k, 1] = float(parts[3])
                i += 1

            # sort by atom id so both frames align
            order = np.argsort(ids)
            results[ts] = pos[order]
            print(f"  Parsed timestep {ts}: {num_atoms} atoms")

            if len(results) == 2:
                break
        else:
            i += 1

    if t0 not in results:
        raise RuntimeError(f"Timestep {t0} not found in {filepath}")
    if t50 not in results:
        raise RuntimeError(f"Timestep {t50} not found in {filepath}")

    return results[t0], results[t50], box


# ── spatial Gaussian smoothing on a point cloud ────────────────────────────
def spatial_gaussian_smooth(pos, values, sigma, r_cut):
    """Vectorised Gaussian smooth using scipy sparse distance matrix."""
    print(f"  Building KD-tree and sparse distance matrix (r_cut={r_cut})…")
    tree = cKDTree(pos)
    D = tree.sparse_distance_matrix(tree, max_distance=r_cut,
                                    output_type="coo_matrix")
    W = D.copy()
    W.data = np.exp(-0.5 * (W.data / sigma) ** 2)
    W_csr = W.tocsr()
    row_sum = np.asarray(W_csr.sum(axis=1)).ravel()
    smoothed = W_csr.dot(values) / row_sum
    return smoothed.astype(np.float32)


# ── main ───────────────────────────────────────────────────────────────────
def main():
    print(f"Parsing '{DUMP_FILE}' …")
    pos0, pos50, box = parse_two_frames(DUMP_FILE, TARGET_T0, TARGET_T50)

    xlo, xhi, ylo, yhi = box
    Lx = xhi - xlo
    Ly = yhi - ylo
    print(f"  Box: Lx={Lx:.3f}  Ly={Ly:.3f}")

    # ── squared displacement with periodic boundary conditions ──
    print("Computing squared displacements (PBC) …")
    delta = pos50 - pos0
    delta[:, 0] -= Lx * np.round(delta[:, 0] / Lx)
    delta[:, 1] -= Ly * np.round(delta[:, 1] / Ly)
    sq_disp = (delta ** 2).sum(axis=1)

    print(f"  MSD = {sq_disp.mean():.4f}  |  max = {sq_disp.max():.4f}")

    # ── spatial Gaussian smoothing ──────────────────────────────
    print(f"Applying spatial Gaussian smoothing (sigma={SIGMA}, r_cut={R_SMOOTH}) …")
    labels = spatial_gaussian_smooth(pos0, sq_disp, SIGMA, R_SMOOTH)

    # ── apply natural logarithm to smoothed displacements ────────
    labels = np.log(labels + 1e-8)

    np.save(OUT_FILE, labels)
    print(f"\nSaved {OUT_FILE}  shape={labels.shape}  dtype={labels.dtype}")
    print(f"  label stats: mean={labels.mean():.4f}  std={labels.std():.4f}")


if __name__ == "__main__":
    main()
