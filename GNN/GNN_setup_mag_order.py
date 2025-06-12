from pymatgen.core import Structure
import pandas as pd
import os
import shutil




# Directories
root_dir = '/home/course2024/Bachelor_thesis_2025/POSCARs'
target_poscar_dir = '/home/course2024/Bachelor_thesis_2025/GNN/cgcnn_mag_order/data/poscar'
os.makedirs(target_poscar_dir, exist_ok=True)

# Data holders
id_list = []
filename_lookup = []  # For debugging or mapping back if needed
mag_order_map = {}

i = 0

# Pass 1: collect POSCARs and magnetic moments
for folder_name in sorted(os.listdir(root_dir)):
    folder_path = os.path.join(root_dir, folder_name)
    if os.path.isdir(folder_path):
        for filename in sorted(os.listdir(folder_path)):
            if filename.startswith('POSCAR'):
                try:
                    poscar_path = os.path.join(folder_path, filename)
                    base_key = filename.replace('POSCAR_', '').replace('POSCAR', '').split('.')[0]
                    full_name = f"{folder_name}_{filename}".replace(" ", "").replace("/", "_")

                    i += 1
                    if i <= 2:
                        print(full_name)

                    # Save as idX
                    current_id = f"id{len(id_list)}"
                    id_list.append(current_id)
                    filename_lookup.append((full_name))

                    # Copy POSCAR to poscar dir with variant name (you can use current_id instead if needed)
                    new_poscar_path = os.path.join(target_poscar_dir, full_name)
                    shutil.copy(poscar_path, new_poscar_path)

                except Exception as e:
                    print(f"Could not process {poscar_path}: {e}")

i = 0

# Pass 2: Parse 
for folder_name in sorted(os.listdir(root_dir)):
    folder_path = os.path.join(root_dir, folder_name)
    ground_log_path = os.path.join(folder_path, 'ground_log_sorted')
    if os.path.isfile(ground_log_path):
        with open(ground_log_path) as file:
            for line in file:
                log = line.strip().strip('[]').replace("'", "").split()
                spin_type = 'FM' if log[0] == '1' else 'AFM'
                filename_base = log[3].replace('_optimized', '')
                full_name = f"{folder_name}_{filename_base}_{spin_type}".replace(" ", "").replace("/", "_")

                magnetic_order = log[2]    
                mag_order_map[full_name] = magnetic_order


# Map energy to idX
mag_orders = []
for variant in filename_lookup:
    mag_order = mag_order_map.get(variant)
    mag_orders.append(mag_order)


labels = [0 if mo == 'FM' else 1 for mo in mag_orders]

# Final dataframe
df = pd.DataFrame({
    'id': id_list,
    'mag_order': labels,
})
# One-hot encode: 0 for FM, 1 for AFM/FIM

# Save CSV 
csv_path = "/home/course2024/Bachelor_thesis_2025/GNN/cgcnn_mag_order/data/id_prop.csv"
print('---- Writing to CSV ----')
df.to_csv(csv_path, index=False, header=False)
print(f'---- Data written to {csv_path} ----')
print(df.head())
