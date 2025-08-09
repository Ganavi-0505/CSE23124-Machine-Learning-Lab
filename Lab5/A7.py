import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

df = pd.read_excel("lab_5_dataset.xlsx")
target_col = 'failure' if 'failure' in df.columns else df.columns[-1]
X = df.drop(columns=[target_col])

distortions = []
for k in range(2, 20):
    km = KMeans(n_clusters=k, random_state=42, n_init="auto").fit(X)
    distortions.append(km.inertia_)

plt.plot(range(2, 20), distortions, marker='o')
plt.xlabel("k")
plt.ylabel("Distortion")
plt.title("Elbow Method")
plt.savefig("A7_elbow_plot.png")
print("Complete")