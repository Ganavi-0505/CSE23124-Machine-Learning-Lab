import pandas as pd
import numpy as np

# Load the file
df = pd.read_excel("lab_5_dataset.xlsx")

# Step 1: Replace empty strings with NaN
df.replace(r'^\s*$', np.nan, regex=True, inplace=True)

# Step 2: Drop completely empty columns
df.dropna(axis=1, how='all', inplace=True)

# Step 3: Remove non-numeric identifier columns
id_cols = ['date', 'serial_number', 'model', 'datacenter', 
           'cluster_id', 'vault_id', 'pod_id', 'pod_slot_num', 'is_legacy_format']
df.drop(columns=[c for c in id_cols if c in df.columns], inplace=True, errors='ignore')

# Step 4: Convert all remaining columns to numeric
df = df.apply(pd.to_numeric, errors='coerce')

# Step 5: Drop rows with NaN values (you can also use fillna)
df.dropna(axis=0, how='any', inplace=True)

# Step 6: Remove duplicates
df.drop_duplicates(inplace=True)

print("Cleaned shape:", df.shape)
print(df.head())
