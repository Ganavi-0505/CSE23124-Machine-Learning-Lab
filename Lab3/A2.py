import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_excel("lab_3_dataset.xlsx")

feature = 'capacity_bytes'
data = df[feature].dropna()

# Histogram
counts, bins = np.histogram(data, bins=10)
plt.hist(data, bins=10, edgecolor='black')
plt.title(f"Histogram of {feature}")
plt.xlabel(feature)
plt.ylabel("Frequency")
plt.show()
plt.savefig("Histogram A2.png")
print("\n✅ Plot saved as Histogram A2.png")

# Mean and Variance
mean_val = data.mean()
variance_val = data.var()
print(f"Mean: {mean_val:.2e}")
print(f"Variance: {variance_val:.2e}")
