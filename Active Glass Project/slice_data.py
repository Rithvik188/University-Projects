import numpy as np
import os

def extract_2k(input_file, output_file, num_atoms=2000):
    with open(input_file, 'r') as f:
        lines = f.readlines()

    out_lines = []
    
    header_end = 0
    atoms_start = 0
    x_line_idx = y_line_idx = z_line_idx = -1
    
    for i, line in enumerate(lines):
        if 'atoms' in line:
            lines[i] = f"{num_atoms} atoms\n"
        elif 'xlo xhi' in line:
            x_line_idx = i
        elif 'ylo yhi' in line:
            y_line_idx = i
        elif 'zlo zhi' in line:
            z_line_idx = i
        elif line.startswith('Atoms'):
            atoms_start = i
            break
            
    atom_data = []
    for i in range(atoms_start + 2, len(lines)):
        if lines[i].strip():
            parts = lines[i].split()
            if len(parts) >= 5:
                atom_data.append(parts)
            if len(atom_data) == num_atoms:
                break
                
    # Calculate bounding box
    xs = [float(p[2]) for p in atom_data]
    ys = [float(p[3]) for p in atom_data]
    
    x_min, x_max = min(xs), max(xs)
    y_min, y_max = min(ys), max(ys)
    
    # Add small padding to prevent boundary errors
    padding = 0.5
    x_min -= padding
    x_max += padding
    y_min -= padding
    y_max += padding
    
    dx = (x_max + x_min) / 2
    dy = (y_max + y_min) / 2
    
    box_lx = x_max - x_min
    box_ly = y_max - y_min
    
    lines[x_line_idx] = f"{-box_lx/2:.6f} {box_lx/2:.6f} xlo xhi\n"
    lines[y_line_idx] = f"{-box_ly/2:.6f} {box_ly/2:.6f} ylo yhi\n"
    
    with open(output_file, 'w') as f:
        for i in range(atoms_start + 2):
            f.write(lines[i])
            
        for parts in atom_data:
            # Shift coordinates so centroid is around 0
            parts[2] = f"{float(parts[2]) - dx:.6f}"
            parts[3] = f"{float(parts[3]) - dy:.6f}"
            f.write(' '.join(parts) + '\n')

if __name__ == '__main__':
    script_dir = os.path.dirname(os.path.abspath(__file__))
    input_file = os.path.join(script_dir, 'input.file')
    output_file = os.path.join(script_dir, 'test_2k.data')
    extract_2k(input_file, output_file)
    print("Extraction and re-centering complete.")
