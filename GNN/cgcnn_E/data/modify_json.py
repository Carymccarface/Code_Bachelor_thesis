import json
from pymatgen.core import Element

# Load your original atom_init.json
with open("atom_init.json", "r") as f:
    data = json.load(f)

# Convert keys from element symbols to atomic numbers
converted = {str(Element(el).Z): value for el, value in data.items()}

# Save the corrected version
with open("atom_init.json", "w") as f:
    json.dump(converted, f, indent=4)
print('Success')