import pandas as pd
import numpy as np

# Load the Excel file
df = pd.read_excel("lab_3_dataset.xlsx")

# Preview first few rows to identify structure
print("Dataset Preview:\n", df.head())

# Assume the label column is named 'label' or 'class' and features are numeric columns
label_col = 'model'

feature_cols = df.select_dtypes(include=[np.number]).columns.tolist()

# Get two class labels
class_labels = df[label_col].unique()
class1, class2 = class_labels[0], class_labels[1]

# Filter data for each class
data1 = df[df[label_col] == class1][feature_cols]
data2 = df[df[label_col] == class2][feature_cols]

# Compute centroids
centroid1 = data1.mean().values
centroid2 = data2.mean().values

# Compute intraclass spreads (standard deviation for each class)
spread1 = data1.std().values
spread2 = data2.std().values

# Mask to exclude NaNs from both centroids
valid_mask = ~np.isnan(centroid1) & ~np.isnan(centroid2)
filtered_centroid1 = centroid1[valid_mask]
filtered_centroid2 = centroid2[valid_mask]
# Compute interclass distance (using Euclidean)
interclass_distance = np.linalg.norm(filtered_centroid1 - filtered_centroid2)


# Printing results
print(f"\nSelected Classes: {class1} and {class2}\n")
print(f"Centroid of class {class1}:\n{centroid1}")
print(f"Centroid of class {class2}:\n{centroid2}")
print(f"\nIntraclass Spread of class {class1}:\n{spread1}")
print(f"Intraclass Spread of class {class2}:\n{spread2}")
print(f"\nInterclass Distance between class {class1} and {class2}: {interclass_distance:.4f}")
