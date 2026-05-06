import sys

def fix_z_coords(filename):
    with open(filename, 'r') as f:
        lines = f.readlines()
        
    with open(filename, 'w') as f:
        in_atoms = False
        for line in lines:
            if line.startswith('Atoms'):
                in_atoms = True
                f.write(line)
            elif in_atoms and line.strip() and len(line.split()) >= 5:
                parts = line.split()
                # 5th column is z (index 4)
                parts[4] = '0.0'
                f.write(' '.join(parts) + '\n')
            elif line.startswith('Velocities') or line.startswith('Masses') or line.startswith('Pair Coeffs'):
                in_atoms = False
                f.write(line)
            else:
                f.write(line)

if __name__ == '__main__':
    fix_z_coords('test_2k.data')
    print("Fixed z coordinates.")
