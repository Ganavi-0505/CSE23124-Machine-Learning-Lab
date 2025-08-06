import numpy as np
import matplotlib.pyplot as plt

np.random.seed(0)
train_X = np.random.uniform(1, 10, size=(20, 2))
train_y = np.array([0 if x[0]+x[1] < 11 else 1 for x in train_X])  # arbitrary rule

plt.figure()
plt.scatter(train_X[train_y==0, 0], train_X[train_y==0, 1], c='blue', label='Class 0')
plt.scatter(train_X[train_y==1, 0], train_X[train_y==1, 1], c='red', label='Class 1')
plt.xlabel("X")
plt.ylabel("Y")
plt.title("A3: Training Data Scatter Plot")
plt.legend()
plt.show()
plt.savefig("A3.png")