import pandas as pd
import numpy as np
from sklearn.preprocessing import KBinsDiscretizer

# Load dataset
df = pd.read_excel("DataSet2.xlsx")

# Find a numeric column with more than 1 unique value
numeric_cols = df.select_dtypes(include=[np.number]).columns
num_col = None
for col in numeric_cols:
    if df[col].nunique() > 1:
        num_col = col
        break

# Binning
disc = KBinsDiscretizer(n_bins=3, encode='ordinal', strategy='uniform')
df[num_col] = disc.fit_transform(df[[num_col]]).astype(int)

# Count values per class
class_counts = df[num_col].value_counts().sort_index()

print(f"Binning of column '{num_col}' (bins=3):")
for cls, count in class_counts.items():
    print(f"Class {cls}  →  {count} values")
