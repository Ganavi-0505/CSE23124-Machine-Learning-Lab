import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split

df = pd.read_excel('DataSet2.xlsx')

X = df[['smart_1_raw', 'smart_5_raw']]
y = df['failure']

X = X.dropna()
y = y.loc[X.index]

X_train, _, y_train, _ = train_test_split(X, y, stratify=y, test_size=0.3, random_state=42)

x_min, x_max = 0, 1000
y_min, y_max = 0, 1000
xx, yy = np.meshgrid(np.arange(x_min, x_max, 10),
                     np.arange(y_min, y_max, 10))
test_points = np.c_[xx.ravel(), yy.ravel()]

knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)
test_pred = knn.predict(test_points)

plt.figure()
plt.scatter(test_points[:, 0], test_points[:, 1], c=test_pred, cmap='coolwarm', alpha=0.2)
plt.scatter(X_train.iloc[:, 0], X_train.iloc[:, 1], c=y_train, cmap='coolwarm', edgecolor='k')
plt.xlabel("smart_1_raw")
plt.ylabel("smart_5_raw")
plt.title("A6: Dataset2 SMART Attribute Classification with k=5")
plt.show()
plt.savefig("A6_1.png")