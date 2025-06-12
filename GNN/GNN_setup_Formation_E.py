from pymatgen.core import Structure
import pandas as pd
import os
import shutil
import numpy as np
import subprocess

form  = {
"Al"  :  -3.7155945,
"As"  :  -4.6496745,
"B"   :  -6.70374805,
"Bi"  :  -3.82053325,
"Br"  :  -1.634252750,
"C"   :  -9.03417425,
"Cl"  :  -1.8368175,
"F"   :  -1.776911225,
"Ga"  :  -2.909076,
"Ge"  :  -4.47091775,
"I"   :  -1.4635728,
"In"  :  -2.53951710,
"N"   :  -8.3200605,
"O"   :  -4.5362375,
"P"   :  -5.373504125,
"Pb"  :  -3.575652500,
"S"   :  -4.12675781250,
"Sb"  :  -4.1398070,
"Se"  :  -3.49823666666,
"Si"  :  -5.42027,
"Sn"  :  -3.8174223,
"Te"  :  -3.1421034666,
"Tl"  :  -2.200622,
"Ti"  :  -7.773629,
"V"   :  -8.9377110,
"Cr"  :  -9.48089749,
"Mn"  :  -8.89637525,
"Fe"  :  -8.2359165,
"Co"  :  -7.019172, 
"Ni"  :  -5.46768575,
}

def pos_analyser(poscar_path):
    with open(poscar_path) as f:
        for i, lines in enumerate(f.readlines()):
            if i == 2:
                vector = np.array([float(x) for x in lines.split()])
                x_vec = np.linalg.norm(vector)
            if i == 3:
                vector = np.array([float(x) for x in lines.split()])
                y_vec = np.linalg.norm(vector)
            if i == 4:
                vector = np.array([float(x) for x in lines.split()])
                z_vec = np.linalg.norm(vector)
            if i == 5:
                elements = np.array([str(x) for x in lines.split()])
            if i == 6:
                sum_atom = np.sum(np.array([int(x) for x in lines.split()]))
                num_atom = np.array([int(x) for x in lines.split()])
    return (elements, sum_atom, num_atom)

def total_energy(outcar_path):
    try:
        E = float(subprocess.check_output(
            f"grep 'sigma->0' '{outcar_path}' | tail -n 1 | awk '{{print $NF}}'", shell=True))
    except Exception:
        E = None
    return E


# Directories
root_dir = '/home/course2024/Bachelor_thesis_2025/POSCARs'
target_poscar_dir = '/home/course2024/Bachelor_thesis_2025/GNN/cgcnn_Formation_E/data/poscar'
os.makedirs(target_poscar_dir, exist_ok=True)

# Data holders
id_list = []
filename_lookup = []  # For debugging or mapping back if needed

E_formation = []


# Pass 1: collect POSCARs and magnetic moments
for folder_name in sorted(os.listdir(root_dir)):
    folder_path = os.path.join(root_dir, folder_name)
    if os.path.isdir(folder_path):
        for filename in sorted(os.listdir(folder_path)):
            if filename.startswith('POSCAR'):
                try:
                    poscar_path = os.path.join(folder_path, filename)
                    base_key = filename.replace('POSCAR_', '').replace('POSCAR', '').split('.')[0]
                    variant_name = f"{folder_name}_{filename}".replace(" ", "").replace("/", "_")

                    # Save as idX
                    current_id = f"id{len(id_list)}"
                    id_list.append(current_id)
                    filename_lookup.append((current_id, variant_name))

                    # Copy POSCAR to poscar dir with variant name (you can use current_id instead if needed)
                    new_poscar_path = os.path.join(target_poscar_dir, variant_name)
                    shutil.copy(poscar_path, new_poscar_path)

                    for f in os.listdir(folder_path):
                        if f.startswith('OUTCAR') and base_key in f:
                            matching_outcar_path = os.path.join(folder_path, f)
                            elements, sum_atom, num_atom = pos_analyser(poscar_path)
                            E_t = total_energy(matching_outcar_path)

                            if isinstance(E_t, float):  # Ensure total_energy succeeded
                                E_form = E_t
                                for i in range(len(elements)):
                                    for j in range(num_atom[i]):
                                        E_form -= form[elements[i]]
                                E_form /= np.sum(num_atom)
                            else:
                                E_form = None
                            E_formation.append(E_form)

                except Exception as e:
                    print(f"Could not process {poscar_path}: {e}")


# Final dataframe
df = pd.DataFrame({
    'id': id_list,
    'Formation_E': E_formation,
})

# Save CSV (no header)
csv_path = "/home/course2024/Bachelor_thesis_2025/GNN/cgcnn_Formation_E/data/id_prop.csv"
print('---- Writing to CSV ----')
df.to_csv(csv_path, index=False, header=False)
print(f'---- Data written to {csv_path} ----')
print(df.head())