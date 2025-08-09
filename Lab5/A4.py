import pandas as pd
from sklearn.cluster import KMeans

df = pd.read_excel("lab_5_dataset.xlsx")
target_col = 'failure' if 'failure' in df.columns else df.columns[-1]
X = df.drop(columns=[target_col])

kmeans = KMeans(n_clusters=2, random_state=42, n_init="auto")
kmeans.fit(X)

print("Cluster Centers:\n", kmeans.cluster_centers_)