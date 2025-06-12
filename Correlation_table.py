import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

# Load dataset
df = pd.read_csv('all_features_spin_fixed.csv')
data = df.drop(columns=['energy', 'tot_mag_mom', 'E_formation'])

# Ensure the target is numerical
if not pd.api.types.is_numeric_dtype(data['mag_order']):
    data['mag_order'] = pd.Categorical(data['mag_order']).codes

# --- STEP 1: Filter columns by label ---
selected_features = [
    'jml_KV', 'jml_GV', 'jml_op_eg', 'jml_Z', 'jml_X',
    'jml_avg_ion_rad', 'jml_cell_3', 'jml_nn_0', 'jml_ddf_0',
    'jml_adfb_0', 'jml_adfa_0', 'jml_mean_chg_0', 'density', 'site #1 min. rel. dist.', 'site #0 min. rel. dist.',
    'site #2 min. rel. dist.', 'site #3 min. rel. dist.', 'Cr0+ - Cr0+ bond #0', 'Cr0+ - Cr0+ bond #1'
]

numeric_cols = data.select_dtypes(include='number').columns


# Filter columns that contain any of the keywords
filtered_cols = [col for col in numeric_cols if any(kw in col for kw in selected_features)]
#filtered_cols = [col for col in numeric_cols if col in selected_features]


# Always include target for correlation
if 'mag_order' not in filtered_cols:
    filtered_cols.append('mag_order')

# --- STEP 2: Compute correlation with target ---
corr_matrix = data[filtered_cols].corr()
target_corr = corr_matrix['mag_order'].drop('mag_order')  # Drop self-correlation
target_corr_abs_sorted = target_corr.abs().sort_values(ascending=False)

# --- STEP 3: Plot as 1D bar chart ---
plt.figure(figsize=(10, 6))
bars = plt.barh(target_corr_abs_sorted.index, target_corr_abs_sorted.values, color='firebrick')
plt.xlabel('Absolute Correlation with mag_order')
plt.title('Feature Correlations with mag_order')
plt.gca().invert_yaxis()  # Highest correlations on top
plt.grid(axis='x')

# Annotate bars with actual values
for bar in bars:
    width = bar.get_width()
    plt.text(width + 0.01, bar.get_y() + bar.get_height()/2,
             f'{width:.2f}', va='center', fontsize=9)

# Save figure
folder_path = 'heatmaps'
os.makedirs(folder_path, exist_ok=True)
plt.tight_layout()
plt.savefig(f'{folder_path}/barplot_correlations_with_mag_order.png', dpi=300)
plt.show()

# # --- STEP 3: Split into chunks for readability ---
# chunk_size = len(target_corr_abs_sorted) // 2 + len(target_corr_abs_sorted) % 2
# chunks = [target_corr_abs_sorted.iloc[i:i+chunk_size] for i in range(0, len(target_corr_abs_sorted), chunk_size)]

# # Plot each chunk
# for i, chunk in enumerate(chunks, 1):
#     plt.figure(figsize=(12, 8))
#     bars = plt.barh(chunk.index, chunk.values, color='skyblue')
#     plt.xlabel('Absolute Correlation with mag_order')
#     plt.title(f'Feature Correlations with mag_order (Part {i})')
#     plt.gca().invert_yaxis()
#     plt.grid(axis='x')

#     # Annotate bars
#     for bar in bars:
#         width = bar.get_width()
#         plt.text(width + 0.01, bar.get_y() + bar.get_height()/2,
#                  f'{width:.2f}', va='center', fontsize=9)

#     # Save each figure
#     folder_path = 'heatmaps'
#     os.makedirs(folder_path, exist_ok=True)
#     plt.tight_layout()
#     plt.savefig(f'{folder_path}/barplot_correlations_with_mag_order_part_{i}.png', dpi=300)
#     plt.show()
