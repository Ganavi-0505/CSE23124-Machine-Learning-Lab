import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import minkowski

df = pd.read_excel("lab_3_dataset.xlsx")
classes = df['model'].unique()[:2]
df = df[df['model'].isin(classes)].dropna(axis=1)

vec1 = df[df['model'] == classes[0]].iloc[0]
vec2 = df[df['model'] == classes[1]].iloc[0]

drop_cols = ['model', 'serial_number', 'date', 'datacenter', 'cluster_id']
vec1 = vec1.drop([col for col in drop_cols if col in vec1.index])
vec2 = vec2.drop([col for col in drop_cols if col in vec2.index])

r_values = list(range(1, 11))
distances = [minkowski(vec1, vec2, p=r) for r in r_values]

plt.plot(r_values, distances, marker='o')
plt.title("Minkowski Distance (r=1 to 10)")
plt.xlabel("Order r")
plt.ylabel("Distance")
plt.grid(True)
plt.show()
plt.savefig("Minkowski.png")
print("\n✅ Plot saved as 'Minkowski.png'")