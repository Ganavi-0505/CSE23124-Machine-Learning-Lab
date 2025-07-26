import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

# Load dataset
df = pd.read_excel("lab_3_dataset.xlsx", sheet_name="Sheet1")
label_col = 'model'
df = df.dropna()
selected_classes = df[label_col].value_counts().index[:2]
df = df[df[label_col].isin(selected_classes)]

X = df.drop(columns=[label_col])
y = df[label_col]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Evaluate accuracy for different k values
accuracies = []
k_range = range(1, 12)
for k in k_range:
    model = KNeighborsClassifier(n_neighbors=k)
    model.fit(X_train, y_train)
    y_k_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_k_pred)
    accuracies.append(acc)

# Plot
plt.figure(figsize=(8, 5))
plt.plot(k_range, accuracies, marker='o')
plt.title('Accuracy vs k in k-NN')
plt.xlabel('k')
plt.ylabel('Accuracy')
plt.grid(True)
plt.tight_layout()
plt.savefig("A8_accuracy_plot.png")
plt.show()
