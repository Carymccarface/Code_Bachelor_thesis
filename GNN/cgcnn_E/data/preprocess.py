import json
import numpy as np
from pymatgen.core import Element

# Define the size of the atomic feature vector (e.g., 92 dimensions)
atom_feature_length = 92

# Get all elements
elements = list(Element)

# Create a dictionary to store atomic features
atom_features = {}

# Generate random features for each element
for element in elements:
    # Each element will have a random feature vector of 'atom_feature_length'
    atom_features[element.symbol] = np.random.rand(atom_feature_length).tolist()

# Write the atom features to a JSON file
with open('atom_init.json', 'w') as f:
    json.dump(atom_features, f, indent=4)

print("atom_init.json has been created successfully!")
