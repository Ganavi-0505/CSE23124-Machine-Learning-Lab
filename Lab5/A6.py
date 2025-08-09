import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score

df = pd.read_excel("lab_5_dataset.xlsx")
target_col = 'failure' if 'failure' in df.columns else df.columns[-1]
X = df.drop(columns=[target_col])

k_values = range(2, 10)
sil_scores, ch_scores, db_scores = [], [], []

for k in k_values:
    km = KMeans(n_clusters=k, random_state=42, n_init="auto").fit(X)
    sil_scores.append(silhouette_score(X, km.labels_))
    ch_scores.append(calinski_harabasz_score(X, km.labels_))
    db_scores.append(davies_bouldin_score(X, km.labels_))

plt.plot(k_values, sil_scores, label="Silhouette")
plt.plot(k_values, ch_scores, label="Calinski-Harabasz")
plt.plot(k_values, db_scores, label="Davies-Bouldin")
plt.xlabel("k")
plt.ylabel("Score")
plt.legend()
plt.savefig("A6.png")
print("Completed")