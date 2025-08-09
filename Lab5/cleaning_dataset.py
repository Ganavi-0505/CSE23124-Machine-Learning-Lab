import pandas as pd
import numpy as np

# Load the file
df = pd.read_excel("lab_5_dataset.xlsx")

# Step 1: Replace empty strings with NaN
df.replace(r'^\s*$', np.nan, regex=True, inplace=True)

# Step 2: Drop completely empty columns
df.dropna(axis=1, how='all', inplace=True)

# Step 3: Remove non-numeric identifier columns (if present)
id_cols = ['date', 'serial_number', 'model', 'datacenter', 
           'cluster_id', 'vault_id', 'pod_id', 'pod_slot_num', 'is_legacy_format']
df.drop(columns=[c for c in id_cols if c in df.columns], inplace=True, errors='ignore')

# Step 4: Drop columns with less than 50% non-null values
threshold = 0.5 * len(df)
df.dropna(axis=1, thresh=threshold, inplace=True)

# Step 5: Convert all remaining columns to numeric (coerce errors to NaN)
df = df.apply(pd.to_numeric, errors='coerce')

# Step 6: Data imputation
num_cols = df.select_dtypes(include=[np.number]).columns

for col in num_cols:
    if df[col].isnull().sum() > 0:
        # Detect outliers using IQR
        q1, q3 = df[col].quantile(0.25), df[col].quantile(0.75)
        iqr = q3 - q1
        lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        has_outliers = ((df[col] < lower) | (df[col] > upper)).any()

        if has_outliers:
            df[col].fillna(df[col].median(), inplace=True)
            print(f"{col}: Filled missing with Median (outliers detected)")
        else:
            df[col].fillna(df[col].mean(), inplace=True)
            print(f"{col}: Filled missing with Mean (no outliers)")
    else:
        print(f"{col}: No missing values — no imputation needed.")

# Step 7: Remove duplicates
df.drop_duplicates(inplace=True)

# Step 8: Save cleaned dataset OVER the original file
df.to_excel("lab_5_dataset.xlsx", index=False)

print("✅ Dataset cleaned and saved as lab_5_dataset.xlsx")
print("Cleaned shape:", df.shape)
print("Remaining missing values per column:\n", df.isnull().sum())