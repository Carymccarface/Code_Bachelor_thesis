import pandas as pd
import os
import seaborn as sns
import matplotlib.pyplot as plt

# Load the dataset
data = pd.read_csv('all_features_pp_outcar.csv')  # Update path as needed

# Ensure the target is numerical; skip if already numeric
if not pd.api.types.is_numeric_dtype(data['mag_mom']):
    data['mag_mom'] = pd.Categorical(data['mag_mom']).codes

# Select all numeric columns including the new target
numeric_cols = data.select_dtypes(include=['number']).columns.tolist()
print(f"Total number of numeric features: {len(numeric_cols)}")

# Create correlation matrix
corr_matrix = data[numeric_cols].corr()

# Show only the top-k features most correlated with the target 'mag_mom'
target_corr = corr_matrix['mag_mom'].abs().sort_values(ascending=False)

# Choose how many top features you want to display
top_k = 10
top_features = target_corr.head(top_k).index
top_corr_values = target_corr.head(top_k).values

# Subset correlation matrix to include only top features
top_corr_matrix = corr_matrix.loc[top_features, top_features]

# Truncate column names for cleaner plot display
short_labels = [name[:15] + '…' if len(name) > 15 else name for name in top_corr_matrix.columns]
top_corr_matrix_short = top_corr_matrix.copy()
top_corr_matrix_short.columns = short_labels
top_corr_matrix_short.index = short_labels

# Plot heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(top_corr_matrix_short, annot=True, cmap='coolwarm', fmt=".2f")
plt.title(f'Top {top_k} Feature Correlations with Magnetic Moment (mag_mom)')

# Save plot
folder_path = 'heatmaps'
os.makedirs(folder_path, exist_ok=True)
save_path = f'{folder_path}/top_k={top_k}_correlations_mag_mom.png'
plt.savefig(save_path, dpi=300, bbox_inches='tight')
plt.show()

# ----------------------------
# Print Top-k Descriptors Info
# ----------------------------

# Build and display a DataFrame with feature info
feature_info = pd.DataFrame({
    'Feature Name': top_features,
    'Correlation with Magnetic Moment': top_corr_values
})

print("\nTop K Descriptors Correlated with Magnetic Moment:\n")
print(feature_info.to_string(index=False))
