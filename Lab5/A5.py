import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score

df = pd.read_excel("lab_5_dataset.xlsx")
target_col = 'failure' if 'failure' in df.columns else df.columns[-1]
X = df.drop(columns=[target_col])

kmeans = KMeans(n_clusters=2, random_state=42, n_init="auto")
labels = kmeans.fit_predict(X)

print("Silhouette Score:", silhouette_score(X, labels))
print("Calinski-Harabasz Score:", calinski_harabasz_score(X, labels))
print("Davies-Bouldin Index:", davies_bouldin_score(X, labels))