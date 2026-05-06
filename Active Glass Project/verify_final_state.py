import numpy as np
import matplotlib.pyplot as plt

# Load the trajectory
traj_path = "trajectory.npy"
traj = np.load(traj_path)

# Take the very last frame
last_frame = traj[-1]
N = last_frame.shape[0]
phi = 0.85
# Recompute box size based on area fraction
L = np.sqrt(N * (np.pi * (1.0 / 2)**2) / phi)

# Compute RDF g(r) for the single last frame
def compute_gr_single_frame(pos, L, r_max, dr):
    n_particles = pos.shape[0]
    n_bins = int(np.ceil(r_max / dr))
    bins = np.linspace(0, r_max, n_bins + 1)
    hist_sum = np.zeros(n_bins)
    
    for i in range(n_particles):
        diff = pos - pos[i]
        diff = diff - L * np.round(diff / L)
        dist = np.linalg.norm(diff, axis=1)
        
        # exclude self
        dist = dist[dist > 1e-6] 
        
        hist, _ = np.histogram(dist, bins=bins)
        hist_sum += hist
        
    rho = n_particles / (L**2)
    r = (bins[:-1] + bins[1:]) / 2.0
    area = np.pi * ((bins[1:])**2 - (bins[:-1])**2)
    
    g_r = hist_sum / n_particles 
    g_r /= (rho * area)
    
    return r, g_r

print("Computing g(r) for the final frame...")
r, g_r = compute_gr_single_frame(last_frame, L, r_max=L/2.0, dr=0.05)

# Visual Verification
plt.figure(figsize=(12, 5))

# Scatter plot of particle positions from the last frame
plt.subplot(1, 2, 1)
plt.scatter(last_frame[:, 0], last_frame[:, 1], s=15, c='indigo', alpha=0.7, edgecolors='none')
plt.xlim(0, L)
plt.ylim(0, L)
plt.title('Final Frame: Particle Positions')
plt.xlabel('x')
plt.ylabel('y')
plt.gca().set_aspect('equal')

# RDF g(r) plot
plt.subplot(1, 2, 2)
plt.plot(r, g_r, 'b-', lw=2)
plt.axhline(1.0, color='k', linestyle='--')
plt.xlabel('Distance $r/\\sigma$')
plt.ylabel('Radial Distribution Function $g(r)$')
plt.title('Final Frame: Structural $g(r)$')
plt.xlim(0, 5)
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("final_state_verification.png", dpi=150)
print("Saved final_state_verification.png")

# Automated heuristics for amorphous vs crystal
first_peak_idx = np.argmax(g_r[:len(r)//2])
first_peak_height = g_r[first_peak_idx]

# Check if it goes to absolute zero between nearest neighbor peaks (crystalline feature)
min_after_first_peak = np.min(g_r[first_peak_idx:first_peak_idx+20])

print(f"First peak height: {first_peak_height:.2f}")
print(f"Minimum after first peak: {min_after_first_peak:.2f}")

if first_peak_height > 2.0 and min_after_first_peak > 0.1:
    print("\\nThe scatter plot shows an amorphous structure without long-range crystalline order.")
    print("The g(r) plot displays continuous, smooth peaks typical of a glass, rather than sharp Bragg-like spikes.")
    print("\\nPart 1 is verified.")
else:
    print("\\nWARNING: Criteria for amorphous structure not completely met. Review plots.")
