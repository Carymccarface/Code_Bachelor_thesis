import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import os

# Load the dataset
df = pd.read_csv('all_features_pp_spin_fix_nonFM.csv')  # Update path if needed
data = df.drop(columns=['energy', 'encoded', 'E_formation', 'mag_order'])

# Rename 'encoded' to 'magnetic_order'
#data = data.rename(columns={'encoded': 'magnetic_order'})

# Ensure the target is numerical
if not pd.api.types.is_numeric_dtype(data['tot_mag_mom']):
    data['tot_mag_mom'] = pd.Categorical(data['tot_mag_mom']).codes

# Select numeric columns
numeric_cols = data.select_dtypes(include=['number']).columns.tolist()

# Compute correlation matrix
corr_matrix = data[numeric_cols].corr()

# Correlation with target
target_corr = corr_matrix['tot_mag_mom'].abs().sort_values(ascending=False)

# Select top-k features
top_k = 10
top_features = target_corr.head(top_k).index
top_corr_values = target_corr.head(top_k).values

# Extract correlation submatrix
top_corr_matrix = corr_matrix.loc[top_features, top_features]

# Truncate names for heatmap labels
short_labels = [name[:15] + '…' if len(name) > 15 else name for name in top_corr_matrix.columns]
top_corr_matrix_short = top_corr_matrix.copy()
top_corr_matrix_short.columns = short_labels
top_corr_matrix_short.index = short_labels

# Plot the heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(top_corr_matrix_short, annot=True, cmap='coolwarm', fmt=".2f")
plt.title(f'Top {top_k} Feature Correlations with total magnetic moment')

# Save heatmap
folder_path = 'heatmaps'
os.makedirs(folder_path, exist_ok=True)
plt.savefig(f'{folder_path}/top_k={top_k}_correlations_non.png', dpi=300, bbox_inches='tight')
plt.show()

# ----------------------------
# Print Top-k Descriptors Info
# ----------------------------

# Build and display a DataFrame with feature info
feature_info = pd.DataFrame({
    'Feature Name': top_features,
    'Correlation with tot_mag_mom': top_corr_values
})

print("\nTop K Descriptors Correlated with total magnetic moment:\n")
print(feature_info.to_string(index=False))
