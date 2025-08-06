import pandas as pd
from sklearn.linear_model import LinearRegression

# Load the dataset
df = pd.read_excel('Lab_5_Dataset.xlsx')

# Inspect columns
print("Columns in dataset:", df.columns)

# Replace 'NumericFeature' and 'Target' with actual column names with numeric data
X_train = df[['NumericFeature']]  # Single feature (replace with actual)
y_train = df['Target']            # Target variable (replace with actual)

# Train linear regression model
reg = LinearRegression().fit(X_train, y_train)

# Predict on training data
y_train_pred = reg.predict(X_train)

print("=== A1: Linear Regression Model Coefficients ===")
print("Intercept:", reg.intercept_)
print("Coefficients:", reg.coef_)
