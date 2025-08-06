import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier

np.random.seed(0)
train_X = np.random.uniform(1, 10, size=(20, 2))
train_y = np.array([0 if x[0]+x[1] < 11 else 1 for x in train_X])

x_min, x_max = 0, 10
y_min, y_max = 0, 10
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1),
                     np.arange(y_min, y_max, 0.1))
test_points = np.c_[xx.ravel(), yy.ravel()]

for k in [1, 5, 7, 9]:
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(train_X, train_y)
    test_pred = knn.predict(test_points)

    plt.figure()
    plt.scatter(test_points[:, 0], test_points[:, 1], c=test_pred, cmap='coolwarm', alpha=0.2)
    plt.scatter(train_X[:, 0], train_X[:, 1], c=train_y, cmap='coolwarm', edgecolor='k')
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.title(f"A5: Test Grid Classification with k={k}")
    plt.show()
    plt.savefig("A5.png")