import os
import re
import pandas as pd

# Path to the current folder (where this script and CSV files are located)
folder_path = os.path.dirname(os.path.abspath(__file__))

# Regex pattern to capture year and quarter from filenames like data_2020_Q1.csv
pattern = re.compile(r'(\d{4})_Q(\d)')

# Get CSV files, excluding the combined file itself
csv_files = sorted(
    [f for f in os.listdir(folder_path) if f.endswith(".csv") and f != "combined.csv"],
    key=lambda x: tuple(map(int, pattern.search(x).groups()))
    if pattern.search(x) else (9999, 99)
)

# List to store DataFrames
dataframes = []

# Read files in sorted order
for file in csv_files:
    file_path = os.path.join(folder_path, file)
    print(f"Reading {file_path}...")
    df = pd.read_csv(file_path)
    dataframes.append(df)

# Combine all DataFrames (align columns automatically)
combined_df = pd.concat(dataframes, ignore_index=True, sort=False)

# Save to a new CSV file in the same folder
output_path = os.path.join(folder_path, "combined.csv")
combined_df.to_csv(output_path, index=False)

print(f"All CSV files combined by year & quarter into: {output_path}")
