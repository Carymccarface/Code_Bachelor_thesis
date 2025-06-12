import os
import json
from pymatgen.core import Structure
from pymatgen.io.cif import CifWriter

#     # AFM
# poscar_dir = "/home/course2024/Bachelor_thesis_2025/GNN/cgcnn_tot_AFM/data/poscar"
# cif_dir = "/home/course2024/Bachelor_thesis_2025/GNN/cgcnn_tot_AFM/data/"

#     # FM
poscar_dir = "/home/course2024/Bachelor_thesis_2025/GNN/cgcnn_tot_FM/data/poscar"
cif_dir = "/home/course2024/Bachelor_thesis_2025/GNN/cgcnn_tot_FM/data/"

    # ALL MAGMOM
# poscar_dir = "/home/course2024/Bachelor_thesis_2025/GNN/cgcnn_tot_ALL/data/poscar"
# cif_dir = "/home/course2024/Bachelor_thesis_2025/GNN/cgcnn_tot_ALL/data/"


    # E_tot
#poscar_dir = "/home/course2024/Bachelor_thesis_2025/GNN/cgcnn_E/data/poscar"
#cif_dir = "/home/course2024/Bachelor_thesis_2025/GNN/cgcnn_E/data/"


    # E_Form
# poscar_dir = "/home/course2024/Bachelor_thesis_2025/GNN/cgcnn_Formation_E/data/poscar"
# cif_dir = "/home/course2024/Bachelor_thesis_2025/GNN/cgcnn_Formation_E/data/"
 
    # Mag order
# poscar_dir = "/home/course2024/Bachelor_thesis_2025/GNN/cgcnn_mag_order/data/poscar"
# cif_dir = "/home/course2024/Bachelor_thesis_2025/GNN/cgcnn_mag_order/data/"


name_to_id = {}

for i, filename in enumerate(sorted(os.listdir(poscar_dir))):
    if not filename.startswith("."):  # skip hidden files like .DS_Store
        name = os.path.splitext(filename)[0]  # remove extension
        poscar_path = os.path.join(poscar_dir, filename)

        structure = Structure.from_file(poscar_path,)


        # Save as id#.cif (CGCNN format)
        cif_path_id = os.path.join(cif_dir, f"id{i}.cif")
        CifWriter(structure).write_file(cif_path_id)

        # Save as original name (for tracking)
        # cif_path_named = os.path.join(cif_dir, f"{name}.cif")
        # CifWriter(structure).write_file(cif_path_named)

        # Save mapping
        name_to_id[name] = f"id{i}"

# Dump mapping for later use
with open(os.path.join(cif_dir, "name_to_id.json"), "w") as f:
    json.dump(name_to_id, f, indent=2)
print('Done!')