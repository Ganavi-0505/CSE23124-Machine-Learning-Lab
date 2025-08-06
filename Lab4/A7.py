import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.neighbors import KNeighborsClassifier

df = pd.read_excel('DataSet2.xlsx')

X = df[['smart_1_raw', 'smart_5_raw']]
y = df['failure']

X = X.dropna()
y = y.loc[X.index]

X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.3, random_state=42)

param_grid = {'n_neighbors': np.arange(1, 20)}
grid = GridSearchCV(KNeighborsClassifier(), param_grid, cv=5)
grid.fit(X_train, y_train)

print("=== A7: Best k from GridSearchCV ===")
print(f"Best k: {grid.best_params_['n_neighbors']}")
print(f"Best CV Score: {grid.best_score_:.4f}")
