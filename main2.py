from pymatgen.core import Structure
from pymatgen.core.composition import Composition
from pymatgen.io.vasp import Vasprun
from pymatgen.ext.matproj import MPRester
import pandas as pd
import subprocess
import os
import numpy as np
import warnings
from Feature_functions import *  # Imports all defined functions


warnings.filterwarnings("ignore") # disables warnings bloating terminal

#Switch on and off features to extract
# Commented T's stand for test.
# Comment W for working
# Comment E for Error

features_dict_structure = {
    "dimensionality":                       1, #T W Not relevant
    "symmetry":                             1, #T W
    "sites":                                0, 
    "DensityFeatures":                      1, #T W
    "ChemicalOrdering":                     0,      ###Switch to MinimumDistanceNN in Class definition for 2d monolayer
    "MaximumPackingEfficiency":             1, #T W
    "StructuralComplexity":                 1, #T W
    "RadialDistributionFunction":           1, #T W
    "PartialRadialDistributionFunction":    0, #T
    "BranchPointEnergy":                    0, # Not relevant
    "ElectronicRadialDistributionFunction": 0, # only works if the structure has oxidation states
    "multiple_structure":                   1, #W
    "bond_fractions":                       1, #W
    "bag_of_bonds":                         1, #W
    "structural_heterogeneity":             0,      ###does not really work because of the 2D structure, uses vonoroi NN
    "minimum_relative_distances":           1, #W
    "jarvis_cfid":                          1, #W # One of the most important for Energy ML
    "coulomb_matrix":                       1, #W
    "sine_coulomb_matrix":                  1, #W
    "orbital_field_matrix":                 1, #W
    "structure_composition":                1, #W
    "xrd_powder_pattern":                   1, #W
    }

features_dict_structure_vasp = {
     "global_instability_index":             0,     ### Oxidation required Not relevant
     "ewald_energy":                         0,     ### Oxidation required Not relevant
}

features_dict_site = {
    "BondOrientationalParameter":            0,     ### RuntimeError: This structure is pathological, infinite vertex in the Voronoi construction
    "AverageBondLength":                     1, #T, W
    "AverageBondAngle":                      1, #T, W                        
    "ChemicalSRO":                           1, #T, W (multiple NaN values)                              
    "EwaldSiteEnergy":                       0,     ### Error, I think because Oxidation guessed badly                          
    "LocalPropertyDifference":               0,     ### RuntimeError: This structure is pathological, infinite vertex in the Voronoi construction
    "SiteElementalProperty":                 1, #T, W                    
    "SOAP":                                  1, #T, W 
    "AGNIFingerprints":                      1, #T, W                         
    "OPSiteFingerprint":                     1, #T, W                        
    "CrystalNNFingerprint":                  1, #T, W 
    "VoronoiFingerprint":                    0,     ### RuntimeError: This structure is pathological, infinite vertex in the Voronoi construction
    "ChemEnvSiteFingerprint":                0,     ### Doesnt Work. Should work in theory, might be client side error
    "IntersticeDistribution":                1, #T, W 
    "CoordinationNumber":                    1, #T, W 
    "GaussianSymmFunc":                      1, #T, W 
    "GeneralizedRadialDistributionFunction": 1, #T, W Works but outputs a lot of data for each site, might need some work on during ML
    "AngularFourierSeries":                  1, #T, W same as above
    }


#Requires vasprun.xml
features_dict_electronic = {
    "dos":              1, #W
    "band_featurizer":  0,      ### Does not work for our system since it is metallic
    }

#Requires dos=vasprun.complete_dos
features_dict_dos = {
    "SiteDOS":          1, # W
    "DOSFeaturizer":    1, # W
    "DopingFermi":      1, # W
    "Hybridization":    1, # W
    "DosAsymmetry":     1, # W
}
features_dict_composition = {
    "packing":               1, #T W
    "Thermo":                0,      ###E Not relevant
    "ThermoMP":              0,      ###E Not relevant
    "miedema":               1, #W
    "yang":                  1, #W
    "wen":                   1, #W
    "element_property":      1, #W  # Seems important for Energy ML 
    "meredig":               1, #W
    "ElementFraction":       1, #T, W Class to calculate the atomic fraction of each element in a composition.
    "TMetalFraction":        1, #T, W Class to calculate fraction of magnetic transition metals in a composition.
    "Stoichiometry":         1, #T, W Calculate norms of stoichiometric attributes.
    "BandCenter":            1, #T, W Estimation of absolute position of band center using electronegativity.
    "OxidationStates":       0,      ### Useless    ## Works but likely not well # Statistics about the oxidation states for each specie.
    "IonProperty":           1, #T, W ## Works but wants POTCAR&Oxidation State in structure # Ionic property attributes. Similar to ElementProperty.
    "ElectronAffinity":      0,      ### Requires actual oxidation states/Ionic Compound ##Calculate average electron affinity times formal charge of anion elements. 
    "ElectronegativityDiff": 0,      ### Requires Ionic Compund #Features from electronegativity differences between anions and cations. 
    "AtomicOrbitals":        1, #T, W Determine HOMO/LUMO features based on a composition.
    "ValenceOrbital":        1, #T, W Attributes of valence orbital shells
}

#Dont test
features_dict_conversions_struc = {
    "ConversionFeaturizer":         0,
    "StructureToComposition":       0,
    "StructureToIStructure":        0,
    "DictToObject":                 0, #input comp or structure
    "JsonToObject":                 0,
    "StructureToOxidStructure":     0, 
    "PymatgenFunctionApplicator":   0, #T, W ???
    "ASEAtomsToStructure":          0, #T, W
}

#Dont test
features_dict_conversions_comp ={
    "ConversionFeaturizer":         0,
    "StrToComposition":             0,
    "DictToObject":                 0,
    "CompositionToStructureFromMP": 0, # Requires API
}

features_dict_function = {
    "FunctionFeaturizer": 0,        ###Works... I guess? it gives an output, but that is just 'none'
    }

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

################# Function to print features and labels
# This function takes in features and their corresponding labels

def feature_collector(features, labels, row_dict):
    for feature, label in zip(features, labels):
        row_dict[label] = feature

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
################# Function to determine magnetic moment
# 

# 


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


# Root folder that contains composition-named subfolders
root_dir = '/home/course2024/Bachelor_thesis_2025/POSCARs'

# Lists to store data
compositions = []
file_paths = []
Variant = []
vasprun_paths = []
mag_mom = []
tot_mag_mom = []
E_formation=[]

energy_map = {}
mag_order_map = {}

# Walk through all subfolders
for folder_name in os.listdir(root_dir):
    folder_path = os.path.join(root_dir, folder_name)
    if os.path.isdir(folder_path):
        for filename in os.listdir(folder_path):
            if filename.startswith('POSCAR'):  # adapt as needed
                try:
                    full_path = os.path.join(folder_path, filename)
                    poscar_path = os.path.join(folder_path, filename)  # Derive variant key: strip 'POSCAR' 
                    base_key = filename.replace('POSCAR_', '').replace('POSCAR', '').split('.')[0] # e.g. 2_AFM

                # Try to find a vasprun that includes the same variant key
                    for f in os.listdir(folder_path):
                        if f.startswith('vasprun') and base_key in f:
                            matching_vasprun = os.path.join(folder_path, f)
                            break


                    compositions.append(folder_name)
                    file_paths.append(poscar_path)
                    vasprun_paths.append(matching_vasprun)
                                  
                # Try to find an outcar that includes the same variant key
                    for f in os.listdir(folder_path):
                        if f.startswith('OUTCAR') and base_key in f:
                            matching_outcar_path = os.path.join(folder_path, f)
                            moment = moment_calculator(2, matching_outcar_path)
                            tot_mag_mom.append(moment)
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
                
                energy = float(log[1])
                energy_map[full_name] = energy

                magnetic_order = log[2]    
                mag_order_map[full_name] = magnetic_order


# Build a DataFrame
df_materials = pd.DataFrame({
    'composition': compositions,
    'path': file_paths,
    'vasprun_path': vasprun_paths,
    'tot_mag_mom': tot_mag_mom,
    'E_formation': E_formation
})

df_materials['variant'] = df_materials['composition'] + '_' + df_materials['path'].apply(lambda x: os.path.basename(x))
df_materials = df_materials.drop(columns=['composition'])

#Mapping
df_materials['energy'] = df_materials['variant'].map(energy_map)
df_materials['mag_order'] = df_materials['variant'].map(mag_order_map)



#### Featurizing
feature_data = []  # this stays global
Progress = 0

if __name__ == "__main__":

    for i in range(len(df_materials)):

        row_df = df_materials.iloc[i]
        variant = row_df['variant']
        energy = row_df['energy']
        path = row_df['path']
        mag_order = row_df['mag_order']
        tot_mag_mom = row_df['tot_mag_mom']
        E_formation =row_df['E_formation']

        structure = Structure.from_file(path)
        structure = structure.add_oxidation_state_by_guess()
        Sites = structure.sites
        composition = structure.composition
        vasprun = Vasprun(row_df['vasprun_path'], parse_projected_eigen=True)
        dos_vasp=vasprun.complete_dos

        row = {
            "variant": variant,
            "energy": energy,
            'mag_order': mag_order,
            "tot_mag_mom": tot_mag_mom,
            'E_formation': E_formation
        }

        for func in features_dict_structure:
            if features_dict_structure[func]:
                func_ref = globals().get(func)
                if func_ref:
                    features, feature_labels = func_ref(structure)
                    feature_collector(features, feature_labels, row)

        for func in features_dict_structure_vasp:
            if features_dict_structure_vasp[func]:
                func_ref = globals().get(func)
                if func_ref:
                    features, feature_labels = func_ref(vasprun)
                    feature_collector(features, feature_labels, row)

        for func in features_dict_composition:
            if features_dict_composition[func]:
                func_ref = globals().get(func)
                if func_ref:
                    features, feature_labels = func_ref(composition)
                    feature_collector(features, feature_labels, row)

        for func in features_dict_site:
            if features_dict_site[func]:
                func_ref = globals().get(func)
                if func_ref:
                    for idx, site in enumerate(Sites):
                        features, feature_labels = func_ref(structure, idx)
                        feature_labels = [f"{label}_site{idx}" for label in feature_labels]
                        feature_collector(features, feature_labels, row)

        for func in features_dict_electronic:
            if features_dict_electronic[func]:
                func_ref = globals().get(func)
                if func_ref:
                    features, feature_labels = func_ref(vasprun)
                    feature_collector(features, feature_labels, row)

        for func in features_dict_conversions_struc:
            if features_dict_conversions_struc[func]:
                func_ref = globals().get(func)
                if func_ref:
                    features, feature_labels = func_ref(structure)
                    feature_collector(features, feature_labels, row)

        for func in features_dict_conversions_comp:
            if features_dict_conversions_comp[func]:
                func_ref = globals().get(func)
                if func_ref:
                    features, feature_labels = func_ref(composition)
                    feature_collector(features, feature_labels, row)

        for func in features_dict_dos:
            if features_dict_dos[func]:
                func_ref = globals().get(func)
                if func_ref:
                    features, feature_labels = func_ref(dos_vasp)
                    feature_collector(features, feature_labels, row)

#####Vilmer's stuff, should likely be revised to fit the more recent main

    # Function-based features
	  #  for func in features_dict_function.keys():
       	   # if features_dict_function[func] == 1:
              #  features, feature_labels = globals()[func](function)
               # feature_printer(features, feature_labels)

        feature_data.append(row)  # Append only once per structure

        Progress += 1
        Completion = 100*Progress/2760
        if Progress % 5 == 0:
            print(f'----    Progress: {Completion:5.2f} %    ----') #Loadingbar

    print('----    Creating dataframe    ----')
    df_features = pd.DataFrame(feature_data) #Create data frame
    print('----    Done!    ----')

print('----    Writing to csv    ----')
file_name = "all_features_spin_fixed.csv"
df_features.to_csv(file_name, index=False) # Save data as a .csv file
print(f'----    Dataframe written to {file_name}    ----')

print(df_features.head()) # Check header