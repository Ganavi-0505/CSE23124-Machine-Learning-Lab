import pandas as pd
import numpy as np
from sklearn.preprocessing import KBinsDiscretizer

def equal_width_binning(df, column, bins=4):
    discretizer = KBinsDiscretizer(n_bins=bins, encode='ordinal', strategy='uniform')
    df[column] = discretizer.fit_transform(df[[column]]).astype(int)
    return df

# Load dataset
df = pd.read_excel("DataSet2.xlsx")

# Example binning for first numeric column
num_col = df.select_dtypes(include=[np.number]).columns[0]
df_binned = equal_width_binning(df.copy(), num_col, bins=4)

print(f"Binned column '{num_col}':\n", df_binned[num_col].head())