import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score

# =======================
# Step 1: Load cleaned dataset
# =======================
df = pd.read_excel("lab_5_dataset.xlsx")  # Already cleaned by your other script

# Identify target column
target_col = 'failure' if 'failure' in df.columns else df.columns[-1]
X = df.drop(columns=[target_col], errors='ignore')

# =======================
# Step 2: Feature scaling
# =======================
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# =======================
# Step 3: Compute metrics for k=2..9
# =======================
k_values = range(2, 10)
sil_scores, ch_scores, db_scores = [], [], []

for k in k_values:
    km = KMeans(n_clusters=k, random_state=42, n_init="auto").fit(X_scaled)
    labels = km.labels_
    sil_scores.append(silhouette_score(X_scaled, labels))
    ch_scores.append(calinski_harabasz_score(X_scaled, labels))
    db_scores.append(davies_bouldin_score(X_scaled, labels))

# =======================
# Step 4: Plot metrics separately
# =======================
plt.figure(figsize=(15, 4))

# Silhouette
plt.subplot(1, 3, 1)
plt.plot(k_values, sil_scores, marker='o', color='blue')
plt.title("Silhouette Score")
plt.xlabel("k")
plt.ylabel("Score")

# Calinski–Harabasz
plt.subplot(1, 3, 2)
plt.plot(k_values, ch_scores, marker='o', color='orange')
plt.title("Calinski-Harabasz Index")
plt.xlabel("k")
plt.ylabel("Score")

# Davies–Bouldin
plt.subplot(1, 3, 3)
plt.plot(k_values, db_scores, marker='o', color='green')
plt.title("Davies-Bouldin Index")
plt.xlabel("k")
plt.ylabel("Score")

plt.tight_layout()
plt.savefig("A6_metrics_separate.png", dpi=300)
plt.show()

# =======================
# Step 5: Print numeric scores for report
# =======================
print("\nClustering Evaluation Metrics:")
for idx, k in enumerate(k_values):
    print(f"k={k}: Silhouette={sil_scores[idx]:.4f}, "
          f"CH={ch_scores[idx]:.2f}, DB={db_scores[idx]:.4f}")
