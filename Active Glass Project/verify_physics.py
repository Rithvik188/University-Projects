import numpy as np
import matplotlib.pyplot as plt

# Step 1: Data Integrity Check
traj_path = "trajectory.npy"
labels_path = "labels.npy"

traj = np.load(traj_path)
labels = np.load(labels_path)

print("--- Data Integrity Check ---")
print(f"trajectory.npy shape: {traj.shape}")
print(f"labels.npy shape: {labels.shape}")

assert traj.shape[1] == 1024, f"Expected 1024 particles in trajectory, got {traj.shape[1]}"
assert traj.shape[2] == 2, f"Expected 2D trajectory, got {traj.shape[2]}"
assert len(labels) == 1024, f"Expected 1024 labels, got {len(labels)}"

n_frames, n_particles, _ = traj.shape
L = np.sqrt(1024 * (np.pi * (1.0 / 2)**2) / 0.85)

# Step 2: Physics Sanity
def compute_gr(traj, L, r_max, dr):
    """Computes the radial distribution function g(r) for all frames."""
    n_frames, n_particles, _ = traj.shape
    n_bins = int(np.ceil(r_max / dr))
    bins = np.linspace(0, r_max, n_bins + 1)
    hist_sum = np.zeros(n_bins)
    
    for frame in range(n_frames):
        pos = traj[frame]
        
        # Calculate distances with PBC
        # We only really need to go up to L/2 for an isotropic system
        # Doing this naively with a double loop is slow, let's use cdist equivalent
        # with minimum image convention
        for i in range(n_particles):
            diff = pos - pos[i]
            diff = diff - L * np.round(diff / L)
            dist = np.linalg.norm(diff, axis=1)
            
            # exclude self-distance
            dist = dist[dist > 1e-6] 
            
            hist, _ = np.histogram(dist, bins=bins)
            hist_sum += hist
            
    # Normalize
    rho = n_particles / (L**2)
    r = (bins[:-1] + bins[1:]) / 2.0
    area = np.pi * ((bins[1:])**2 - (bins[:-1])**2)
    
    # average over all particles and frames
    g_r = hist_sum / (n_particles * n_frames) 
    g_r /= (rho * area)
    
    return r, g_r

def compute_msd(traj, L):
    """Computes Mean Squared Displacement from unwrapped trajectory."""
    # First, we need to unwrap the trajectory since it was saved folded into the periodic box
    n_frames, n_particles, _ = traj.shape
    unwrapped_traj = np.zeros_like(traj)
    unwrapped_traj[0] = traj[0]
    
    for t in range(1, n_frames):
        # Calculate displacement from last frame
        disp = traj[t] - traj[t-1]
        
        # Apply minimum image convention to find true displacement
        disp = disp - L * np.round(disp / L)
        
        # Add to unwrapped positions
        unwrapped_traj[t] = unwrapped_traj[t-1] + disp
        
    msd = np.zeros(n_frames)
    # Calculate MSD by averaging over all starting frames t0 (time-averaging)
    # For a simple estimate we can just compute displacement from t=0
    # Or for better statistics, average over multiple time origins
    
    # Simple displacement from t=0 (okay for a quick verification script)
    disp_from_0 = unwrapped_traj - unwrapped_traj[0]
    sq_disp = np.sum(disp_from_0**2, axis=2)
    msd = np.mean(sq_disp, axis=1)
    
    return msd

print("\\nComputing g(r)...")
r, g_r = compute_gr(traj, L, r_max=L/2.0, dr=0.05)

print("Computing MSD...")
msd = compute_msd(traj, L)

dt_sim = 0.001
save_freq = 100
time_axis = np.arange(n_frames) * save_freq * dt_sim

# Step 3: Visual Output
plt.figure(figsize=(12, 5))

# Plot g(r)
plt.subplot(1, 2, 1)
plt.plot(r, g_r, 'b-', lw=2)
plt.axhline(1.0, color='k', linestyle='--')
plt.xlabel('Distance $r/\\sigma$')
plt.ylabel('Radial Distribution Function $g(r)$')
plt.title('Structural Verification')
plt.xlim(0, 5)
plt.grid(True, alpha=0.3)

# Find first peak height
first_peak_idx = np.argmax(g_r[:len(r)//2])
first_peak_height = g_r[first_peak_idx]
print(f"\\nFirst peak height of g(r): {first_peak_height:.2f}")

# Plot MSD
plt.subplot(1, 2, 2)
# avoid log(0) at t=0
mask = time_axis > 0
plt.loglog(time_axis[mask], msd[mask], 'r-', lw=2)
plt.xlabel('Time $t$')
plt.ylabel('Mean Squared Displacement $\\langle \\Delta r^2(t) \\rangle$')
plt.title('Dynamical Verification')
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("physics_verification.png", dpi=150)
print("Saved physics_verification.png")

# Verification conclusion
final_msd = msd[-1]
print(f"Final MSD = {final_msd:.4f}")

conclusion = ""
if first_peak_height > 2.5 and final_msd < 1.0:
    conclusion = "GLASSY STATE CONFIRMED: High structural order (g(r) peak) combined with slow/plateauing dynamics (low MSD)."
elif first_peak_height > 2.5 and final_msd > 1.0:
    conclusion = "LIQUID: Structure is solid-like but particles diffuse freely (high MSD)."
elif first_peak_height > 5.0 and final_msd < 0.1:
    conclusion = "CRYSTAL: Extremely high structural order, minimal diffusion."
else:
    conclusion = "UNKNOWN / TRANSITION REGIME based on predefined criteria."

print(f"\\nVERIFICATION RESULT:\\n{conclusion}")
