import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier

# Load cleaned dataset
df = pd.read_excel("DataSet2.xlsx")
if 'date' in df.columns:
    df = df.drop(columns=['date'])

# Use the first 2 features for boundary visualization
target_col = 'failure'
X = df.iloc[:, :2]
y = df[target_col]

# Encode categorical if needed
for col in X.select_dtypes(include=['object','datetime']).columns:
    X[col] = LabelEncoder().fit_transform(X[col].astype(str))

# Train a new tree using only these 2 features
clf = DecisionTreeClassifier(criterion="entropy", max_depth=3, random_state=42)
clf.fit(X, y)

# Meshgrid and prediction
x_min, x_max = X.iloc[:,0].min()-1, X.iloc[:,0].max()+1
y_min, y_max = X.iloc[:,1].min()-1, X.iloc[:,1].max()+1
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1),
                     np.arange(y_min, y_max, 0.1))

Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

plt.contourf(xx, yy, Z, alpha=0.3, cmap='coolwarm')
plt.scatter(X.iloc[:,0], X.iloc[:,1], c=y, edgecolor='k', cmap='coolwarm', s=25)
plt.xlabel(X.columns[0]); plt.ylabel(X.columns[1])
plt.title("Decision Boundary (Decision Tree - 2 Features)")
plt.savefig("A7_decision_boundary.png")
plt.show()