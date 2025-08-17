import pandas as pd
import numpy as np

# Load dataset
df = pd.read_excel('DataSet2.xlsx')

target_column = 'failure'

# Drop identifier columns that should not be considered as features
id_cols = ['serial_number', 'model', 'date', 'datacenter',
           'cluster_id', 'vault_id', 'pod_id', 'pod_slot_num', 'is_legacy_format']

df = df.drop(columns=[c for c in id_cols if c in df.columns], errors='ignore')

# Separate X and y
X = df.drop(columns=[target_column])
y = df[target_column]

# Function to calculate entropy
def calculate_entropy(series):
    probs = series.value_counts(normalize=True).values
    probs = probs[probs > 0]
    return -np.sum(probs * np.log2(probs))


# Gini function
def calculate_gini(series):
    probs = series.value_counts(normalize=True).values
    return 1 - np.sum(probs ** 2)


# Information gain
def information_gain(X, y, feature):
    total_entropy = calculate_entropy(y)
    weighted_entropy = 0
    for value in X[feature].unique():
        subset_y = y[X[feature] == value]
        weighted_entropy += (len(subset_y)/len(y)) * calculate_entropy(subset_y)
    return total_entropy - weighted_entropy

# Compute and display
gains = {feature: information_gain(X, y, feature) for feature in X.columns}
best_feature = max(gains, key=gains.get)

print("A3   Best feature for root node:", best_feature)
print("Information gains:", gains)
