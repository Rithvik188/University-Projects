import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import cKDTree
from tqdm import tqdm
import os

# Parameters
N = 1024
phi = 0.85
v0 = 1.0 # Self-propulsion speed
Dr = 1.0 # Rotational diffusion
dt = 0.001
eq_steps = 100000
prod_steps = 20000
save_freq = 100

sigma = 1.0
epsilon = 1.0

# Calculate box size
particle_area = np.pi * (sigma / 2)**2
total_area = N * particle_area / phi
L = np.sqrt(total_area)

# Initialize positions
grid_size = int(np.ceil(np.sqrt(N)))
spacing = L / grid_size
x, y = np.meshgrid(np.arange(grid_size)*spacing, np.arange(grid_size)*spacing)
pos = np.vstack((x.flatten(), y.flatten())).T[:N]
# Add small noise to prevent aligned artifacts initially
pos += np.random.uniform(-0.1*spacing, 0.1*spacing, size=pos.shape)
pos = np.mod(pos, L)

# Initialize orientations
theta = np.random.uniform(0, 2*np.pi, N)

def compute_forces_kdtree(pos, L, sigma, epsilon):
    """Compute WCA forces with PBC using cKDTree."""
    tree = cKDTree(pos, boxsize=[L, L])
    # WCA cutoff is 2^(1/6) * sigma
    rc = (2**(1/6)) * sigma
    pairs = tree.query_pairs(r=rc, output_type='ndarray')
    
    F = np.zeros_like(pos)
    if len(pairs) > 0:
        i = pairs[:, 0]
        j = pairs[:, 1]
        
        diff = pos[i] - pos[j]
        # Minimum image convention using np.round
        diff = diff - L * np.round(diff / L)
        
        dist_sq = np.sum(diff**2, axis=1)
        dist_sq = np.maximum(dist_sq, 1e-12) # numerical stability against division by zero
        
        r2 = dist_sq
        r6 = (sigma**2 / r2)**3
        r12 = r6**2
        
        # WCA Force magnitude divided by r
        f_mag_over_r = 48 * epsilon * (r12 - 0.5 * r6) / r2
        forces_pair = diff * f_mag_over_r[:, np.newaxis]
        
        # Cap forces to avoid explosions in dense initial state
        f_norms = np.linalg.norm(forces_pair, axis=1)
        max_f = 100.0
        mask = f_norms > max_f
        if np.any(mask):
            forces_pair[mask] = forces_pair[mask] * (max_f / f_norms[mask])[:, np.newaxis]
        
        np.add.at(F, i, forces_pair)
        np.add.at(F, j, -forces_pair)
        
    return F

# Plot initial state
plt.figure(figsize=(8, 8))
plt.scatter(pos[:, 0], pos[:, 1], s=10, c='blue', alpha=0.5)
plt.quiver(pos[:, 0], pos[:, 1], np.cos(theta), np.sin(theta), color='red', scale=30, alpha=0.5)
plt.xlim(0, L)
plt.ylim(0, L)
plt.title(f'Active Glass Initial State (N={N}, $\\phi$={phi})')
plt.gca().set_aspect('equal')
plt.savefig('plot_initial_state.png', dpi=150)
plt.close()
print("Saved plot_initial_state.png")

# Integration variables
unwrapped_pos = pos.copy() # For true displacement calculation to measure propensity

# Pre-allocate trajectory memory
saved_frames = prod_steps // save_freq
trajectory = np.zeros((saved_frames, N, 2))
frame_idx = 0

print("Starting equilibration...")
for step in tqdm(range(eq_steps), desc="Equilibration"):
    F = compute_forces_kdtree(pos, L, sigma, epsilon)
    
    v_active_x = v0 * np.cos(theta)
    v_active_y = v0 * np.sin(theta)
    
    dx = (F[:, 0] + v_active_x) * dt
    dy = (F[:, 1] + v_active_y) * dt
    
    pos[:, 0] += dx
    pos[:, 1] += dy
    
    unwrapped_pos[:, 0] += dx
    unwrapped_pos[:, 1] += dy
    
    pos = np.mod(pos, L)
    
    theta += np.sqrt(2 * Dr * dt) * np.random.randn(N)
    
print("Starting production...")
prod_unwrapped_start = unwrapped_pos.copy()
temperature_sum = 0.0

for step in tqdm(range(prod_steps), desc="Production"):
    F = compute_forces_kdtree(pos, L, sigma, epsilon)
    
    v_active_x = v0 * np.cos(theta)
    v_active_y = v0 * np.sin(theta)
    
    dx = (F[:, 0] + v_active_x) * dt
    dy = (F[:, 1] + v_active_y) * dt
    
    pos[:, 0] += dx
    pos[:, 1] += dy
    
    unwrapped_pos[:, 0] += dx
    unwrapped_pos[:, 1] += dy
    
    pos = np.mod(pos, L)
    
    theta += np.sqrt(2 * Dr * dt) * np.random.randn(N)
    
    if step % save_freq == 0:
        if frame_idx < saved_frames:
            trajectory[frame_idx] = pos.copy()
            frame_idx += 1
        
    # Proxy for instantaneous kinetic temperature
    v_squared = (dx/dt)**2 + (dy/dt)**2
    temperature_sum += np.mean(v_squared) / 2.0  
        
print("Saving output files...")
np.save('trajectory.npy', trajectory)

# Propensity definition: Magnitude of individual displacements over production duration dt_window
displacements = np.linalg.norm(unwrapped_pos - prod_unwrapped_start, axis=1)
np.save('labels.npy', displacements)

T_avg = temperature_sum / prod_steps
print(f"\\n=== Simulation Summary ===")
print(f"System Density (packing fraction): {phi}")
print(f"Average Kinetic Temperature: {T_avg:.4f}")
print("Files generated: trajectory.npy, labels.npy, plot_initial_state.png")
