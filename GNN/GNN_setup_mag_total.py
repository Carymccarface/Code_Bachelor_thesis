from pymatgen.core import Structure
import pandas as pd
import os
import shutil


def moment_calculator(mag_atom_counter, outcar_path):
    """
    Extracts magnetic moments for each magnetic atom from the given OUTCAR file.

    Parameters:
    mag_atom_counter (int): Number of magnetic atoms.
    outcar_path (str or Path): Path to the OUTCAR file.

    Returns:
    list: Magnetic moments for each atom.
    """
    outcar_path = str(outcar_path)  # Ensure compatibility with os.system
    moments = []

    for A in range(mag_atom_counter):
        os.system(f'grep -A {9 + A} "magnetization (x)" {outcar_path} | grep "tot" | tail -1 > out.temp') 
        with open('out.temp') as f:
            line = f.readline()
            moment = float(line.split()[-1])
            #moments.append(moment)

    os.remove('out.temp')
    return moment


# Directories
root_dir = '/home/course2024/Bachelor_thesis_2025/POSCARs'
target_poscar_dir_AFM = '/home/course2024/Bachelor_thesis_2025/GNN/cgcnn_tot_AFM/data/poscar'
os.makedirs(target_poscar_dir_AFM, exist_ok=True)

target_poscar_dir_FM = '/home/course2024/Bachelor_thesis_2025/GNN/cgcnn_tot_FM/data/poscar'
os.makedirs(target_poscar_dir_FM, exist_ok=True)

# Data holders
id_list = []
filename_lookup = []  # For debugging or mapping back if needed
compositions = []
file_paths = []

tot_mag_mom = []
mag_order_map = {}

# Walk through all subfolders
for folder_name in os.listdir(root_dir):
    folder_path = os.path.join(root_dir, folder_name)
    if os.path.isdir(folder_path):
        for filename in os.listdir(folder_path):
            if filename.startswith('POSCAR'):  # adapt as needed
                try:
                    full_path = os.path.join(folder_path, filename)
                    poscar_path = os.path.join(folder_path, filename)  # Derive variant key: strip 'POSCAR' and optional extensions
                    base_key = filename.replace('POSCAR_', '').replace('POSCAR', '').split('.')[0]


                    compositions.append(folder_name)
                    file_paths.append(poscar_path)
                    variant_name = f"{folder_name}_{filename}".replace(" ", "").replace("/", "_")

                    # Save as idX
                    current_id = f"id{len(id_list)}"
                    id_list.append(current_id)
                    filename_lookup.append(variant_name)


                # Try to find an outcar that includes the same variant key
                    for f in os.listdir(folder_path):
                        if f.startswith('OUTCAR') and base_key in f:
                            matching_outcar_path = os.path.join(folder_path, f)

                            moment = moment_calculator(2, matching_outcar_path)
                            tot_mag_mom.append(moment)
                            
                except Exception as e:
                    print(f"Could not read {full_path}: {e}")


# Find energies and magnetic orders and assign to correct POSCAR
for folder_name in os.listdir(root_dir):
    folder_path = os.path.join(root_dir, folder_name)
    ground_log_path = os.path.join(folder_path, 'ground_log_sorted')
    if os.path.isfile(ground_log_path):
        with open(ground_log_path) as file:
            for line in file:
                log = line.strip().strip('[]').replace("'", "").split()
                spin_type = 'FM' if log[0] == '0' else 'AFM'
                filename_base = log[3].replace('_optimized', '')  # e.g., 'POSCAR_1'
                full_name = f"{folder_name}_{filename_base}_{spin_type}" # e.g 'Cr2Al1Al1_POSCAR_1_AFM'

                magnetic_order = log[2]  # Spin post calculations
                mag_order_map[full_name] = magnetic_order


# Build a DataFrame
data = pd.DataFrame({
    'id': id_list,
    'composition': compositions,
    'path': file_paths,
    'tot_mag_mom': tot_mag_mom,
    'Full_name': filename_lookup,
})

# Create variant for mapping
data['variant'] = data['composition'] + '_' + data['path'].apply(lambda x: os.path.basename(x))
data = data.drop(columns=['composition'])

# Mapping
data['mag_order'] = data['Full_name'].map(mag_order_map)

# Where to save
csv_path_FM = "/home/course2024/Bachelor_thesis_2025/GNN/cgcnn_tot_FM/data/id_prop.csv"
csv_path_AFM = "/home/course2024/Bachelor_thesis_2025/GNN/cgcnn_tot_AFM/data/id_prop.csv"

# Split based on 'mag_order'
data_FM = data[data['mag_order'] == 'FM']
data_AFM = data[data['mag_order'] != 'FM']

# Use variant name for filenames
for _, row in data_FM.iterrows():
    shutil.copy(row['path'], os.path.join(target_poscar_dir_FM, row['Full_name']))
for _, row in data_AFM.iterrows():
    shutil.copy(row['path'], os.path.join(target_poscar_dir_AFM, row['Full_name']))


# Drop non ids and targets
data_FM = data_FM.drop(columns = [ 'path',  'Full_name'])
data_AFM = data_AFM.drop(columns = ['path', 'variant', 'Full_name'])

data_FM['id'] = ['id{}'.format(i) for i in range(len(data_FM))]
data_AFM['id'] = ['id{}'.format(i) for i in range(len(data_AFM))]

data_AFM = data_AFM.reset_index(drop=True)
data_FM = data_FM.reset_index(drop=True)

print('FM DF:', data_FM.head())
print('AFM DF:', data_AFM.head())

data_FM = data_FM.drop(columns = ['mag_order', 'variant',])
data_AFM = data_AFM.drop(columns = ['mag_order', ])

# Save to separate CSVs
data_FM.to_csv(csv_path_FM, index=False, header=False)
data_AFM.to_csv(csv_path_AFM, index=False, header=False)


print(f'----    Split complete: {len(data_FM)} FM rows {len(data_AFM)} non-FM rows    ----')

