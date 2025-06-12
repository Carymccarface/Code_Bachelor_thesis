import pandas as pd

data=pd.read_csv("all_features_spin_fixed.csv")
fm_data = data[data['mag_order'] == 'FM']
smallest_rows = fm_data.nsmallest(35, 'E_formation')
print(smallest_rows[['variant', 'E_formation', 'mag_order', 'tot_mag_mom']])