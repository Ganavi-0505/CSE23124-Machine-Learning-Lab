import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

# Load dataset
df = pd.read_excel("lab_3_dataset.xlsx", sheet_name="Sheet1")

# Choose label column
label_col = 'model'

# Drop rows with missing values
df = df.dropna()

# Keep only two classes
selected_classes = df[label_col].value_counts().index[:2]
df = df[df[label_col].isin(selected_classes)]

# Prepare features and labels
X = df.drop(columns=[label_col])
y = df[label_col]

# Split into train and test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Train k-NN classifier with k=3
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train, y_train)
y_pred = knn.predict(X_test)

# Save predictions
pd.DataFrame({"Actual": y_test, "Predicted": y_pred}).to_csv("A7_predictions.csv", index=False)
print("A7: Predictions saved to A7_predictions.csv")