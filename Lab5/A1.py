import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# Load cleaned dataset
df = pd.read_excel("lab_5_dataset.xlsx")

# Select target column
target_col = 'failure' if 'failure' in df.columns else df.columns[-1]
X = df.drop(columns=[target_col])
y = df[target_col]

# Train/test split (1 attribute)
X_train, X_test, y_train, y_test = train_test_split(X.iloc[:, [0]], y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)

y_train_pred = model.predict(X_train)
y_test_pred = model.predict(X_test)

print("Linear Regression with 1 Attribute")
