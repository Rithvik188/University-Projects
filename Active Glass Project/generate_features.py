import numpy as np
from scipy.spatial import Voronoi, cKDTree

def get_voronoi_area(points, L):
    N = len(points)
    areas = np.zeros(N)
    
    # 3x3 tiling for periodic boundaries
    tiled_points = []
    for dx in [-1, 0, 1]:
        for dy in [-1, 0, 1]:
            tiled_points.append(points + np.array([dx*L, dy*L]))
    tiled_points = np.vstack(tiled_points)
    
    vor = Voronoi(tiled_points)
    
    for i in range(N):
        # (0,0) offset is index 4 in our loop
        idx = 4*N + i
        region_idx = vor.point_region[idx]
        region = vor.regions[region_idx]
        
        polygon = vor.vertices[region]
        
        x = polygon[:, 0]
        y = polygon[:, 1]
        area = 0.5 * np.abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))
        areas[i] = area
        
    return areas

def get_q6(points, L, rc=1.5):
    N = len(points)
    tree = cKDTree(points, boxsize=[L, L])
    pairs = tree.query_pairs(r=rc, output_type='ndarray')
    
    q6_complex = np.zeros(N, dtype=complex)
    neighbors_count = np.zeros(N)
    
    if len(pairs) > 0:
        i = pairs[:, 0]
        j = pairs[:, 1]
        
        diff = points[j] - points[i]
        diff = diff - L * np.round(diff / L)
        
        theta = np.arctan2(diff[:, 1], diff[:, 0])
        
        np.add.at(q6_complex, i, np.exp(6j * theta))
        np.add.at(neighbors_count, i, 1)
        
        np.add.at(q6_complex, j, np.exp(6j * (theta + np.pi)))
        np.add.at(neighbors_count, j, 1)
        
    neighbors_count[neighbors_count == 0] = 1
    q6 = np.abs(q6_complex / neighbors_count)
    return q6

def main():
    traj = np.load('trajectory.npy')
    n_frames, N, _ = traj.shape
    phi = 0.85
    L = np.sqrt(N * (np.pi * (1.0 / 2)**2) / phi)
    
    features = np.zeros((n_frames, N, 2))
    
    for frame in range(n_frames):
        pos = traj[frame]
        areas = get_voronoi_area(pos, L)
        q6 = get_q6(pos, L, rc=1.5)
        
        features[frame, :, 0] = areas
        features[frame, :, 1] = q6
        
    np.save('features.npy', features)

if __name__ == '__main__':
    main()
