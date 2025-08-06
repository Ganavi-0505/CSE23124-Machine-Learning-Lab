import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, classification_report

# Load dataset
df = pd.read_excel('DataSet2.xlsx')

# Select features and label
X = df[['smart_1_raw', 'smart_5_raw']]  # choose any 2 numeric SMART attributes
y = df['failure']

# Drop rows with missing values (if any)
X = X.dropna()
y = y.loc[X.index]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.3, random_state=42)

# Fit k-NN
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train, y_train)

# Predict
y_train_pred = knn.predict(X_train)
y_test_pred = knn.predict(X_test)

# Evaluate
print("=== A1: Training Confusion Matrix & Metrics ===")
print(confusion_matrix(y_train, y_train_pred))
print(classification_report(y_train, y_train_pred))

print("\n=== A1: Testing Confusion Matrix & Metrics ===")
print(confusion_matrix(y_test, y_test_pred))
print(classification_report(y_test, y_test_pred))
