import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, classification_report

# Load dataset
df = pd.read_excel("lab_3_dataset.xlsx", sheet_name="Sheet1")
label_col = 'model'
df = df.dropna()
selected_classes = df[label_col].value_counts().index[:2]
df = df[df[label_col].isin(selected_classes)]

X = df.drop(columns=[label_col])
y = df[label_col]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Train k-NN with k=3
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train, y_train)
y_pred = knn.predict(X_test)

# Output confusion matrix and report
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))
