import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import os
import warnings

warnings.filterwarnings("ignore") # disables warnings bloating terminal

# Load data
print('----    Loading data    ----')
data = pd.read_csv("all_features_spin_fixed.csv")
print('----    Done!    ----')

features_num = data.shape[1]
print(f'----    {features_num} features    ----')
print(f'----    Beginning Preprocessing     ----')

# Drop 'variant' if present
data = data.drop(columns=['variant', 'dimensionality'], errors='ignore')

# Drop columns with only a single value for all rows
non_list_columns = data.columns[~data.columns.isin(['tot_mag_mom'])]
filtered = data[non_list_columns]
filtered = filtered.loc[:, filtered.nunique(dropna=False) > 1]
data = pd.concat([filtered, data['tot_mag_mom']], axis=1)

# Replace inf with NaN 
data = data.replace([np.inf, -np.inf], np.nan)

# Remove columns with too many NaNs >65%
nan_threshold = 0.65
nan_fraction = data.isna().mean()
high_nan_cols = nan_fraction[nan_fraction > nan_threshold].index
data = data.drop(columns=high_nan_cols)


# Identify categorical columns
non_numeric_columns = data.select_dtypes(include=['object', 'string']).columns
#print("Non-numeric columns:", non_numeric_columns)

# Columns to label encode
label_cols = [
    'crystal_system', 'site #0 specie', 'site #0 neighbor specie(s)',
    'site #1 specie', 'site #1 neighbor specie(s)', 'site #2 specie',
    'site #2 neighbor specie(s)', 'site #3 specie', 'site #3 neighbor specie(s)',
    'HOMO_character', 'HOMO_element', 'LUMO_character', 'LUMO_element',
    'cbm_character_1', 'cbm_specie_1', 'cbm_location_1',
    'vbm_character_1', 'vbm_specie_1', 'vbm_location_1'
]

# Columns that should be numeric (convert if not already)
numeric_like_cols = [
    'Weight Fraction', 'Atomic Fraction',
    'Average bond angle_site0', 'Average bond angle_site1',
    'Average bond angle_site2', 'Average bond angle_site3',
    'Interstice_dist_mean_site2', 'Interstice_dist_std_dev_site2',
    'Interstice_dist_minimum_site2', 'Interstice_dist_maximum_site2',
    'Interstice_area_mean_site2'
]


# Label encode categorical features
le = LabelEncoder()
for col in label_cols:
    if col in data.columns:
        data[col] = le.fit_transform(data[col].astype(str))


# Redo conversion and fill in case NaNs were introduced (columns with <65% NaNs)
for col in numeric_like_cols:
    if col in data.columns:
        data[col] = pd.to_numeric(data[col], errors='coerce')
# Fill
data = data.fillna(data.mean(numeric_only=True))

# Convert string-number columns to float
for col in numeric_like_cols:
    if col in data.columns:
        data[col] = pd.to_numeric(data[col], errors='coerce')

# Drop columns with only a single value for all rows
non_list_columns = data.columns[~data.columns.isin(['tot_mag_mom'])]
filtered = data[non_list_columns]
filtered = filtered.loc[:, filtered.nunique(dropna=False) > 1]
data = pd.concat([filtered, data['tot_mag_mom']], axis=1)

# Check difference in amount of features
features_num = data.shape[1]

#Encode 0 for FM and 1 for non FM
mag_orders = data['mag_order']
encoding = [0 if mo == 'FM' else 1 for mo in mag_orders]
data['encoded'] = encoding

print(f'----    Done!    ----')
print(f'----    New amount of features {features_num}    ----')

file_name = "all_features_pp_spin_fixed.csv"
print(f'----    Saving as {file_name}    ----')
data.to_csv(file_name, index=False) # Save data as a .csv file
print(f'----    PP data written to {file_name}    ----')

print(data.head()) # Check header

# Split based on 'mag_order'
if 'mag_order' in data.columns:
    data_FM = data[data['mag_order'] == 'FM']
    data_nonFM = data[data['mag_order'] != 'FM']

    # Save to separate CSVs
    data_FM.to_csv("all_features_pp_spin_fix_FM.csv", index=False)
    data_nonFM.to_csv("all_features_pp_spin_fix_nonFM.csv", index=False)

    print(f'----    Split complete: {len(data_FM)} FM rows, {len(data_nonFM)} non-FM rows    ----')
else:
    print("----    Column 'mag_order' not found in data. No split performed.    ----")
