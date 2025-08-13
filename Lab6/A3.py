import pandas as pd
import numpy as np
from collections import Counter
from sklearn.preprocessing import LabelEncoder

def calculate_entropy(y):
    counts = Counter(y)
    total = len(y)
    entropy = 0
    for count in counts.values():
        p = count / total
        entropy -= p * np.log2(p)
    return entropy

def information_gain(X, y, feature):
    total_entropy = calculate_entropy(y)
    values = X[feature].unique()
    weighted_entropy = 0
    for value in values:
        subset_y = y[X[feature] == value]
        weighted_entropy += (len(subset_y) / len(y)) * calculate_entropy(subset_y)
    return total_entropy - weighted_entropy

# Load dataset
df = pd.read_excel("DataSet2.xlsx")
target_col = 'failure'

# Encode categorical features
for col in df.select_dtypes(include=['object']).columns:
    df[col] = LabelEncoder().fit_transform(df[col].astype(str))

X = df.drop(columns=[target_col])
y = df[target_col]

gains = {feature: information_gain(X, y, feature) for feature in X.columns}
best_feature = max(gains, key=gains.get)

print("Best feature for root node:", best_feature)
print("Information gains:", gains)
