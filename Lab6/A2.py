import pandas as pd
import numpy as np
from collections import Counter

def calculate_gini(y):
    counts = Counter(y)
    total = len(y)
    gini = 1
    for count in counts.values():
        p = count / total
        gini -= p ** 2
    return gini

# Load dataset
df = pd.read_excel("DataSet2.xlsx")
target_col = 'failure'
y = df[target_col]

print("Gini Index of dataset:", calculate_gini(y))
