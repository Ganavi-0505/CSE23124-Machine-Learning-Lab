import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

df = pd.read_excel('Lab Session Data.xlsx')
print("Columns in file:", df.columns.tolist())  # debug

# Try matching column names flexibly
# Replace with exact column names if known
for col in df.columns:
    print(f"{col}: {df[col].head(1).values}")

# Replace 'Actual' and 'Predicted' with actual column names
y_true = df.iloc[:, 2]   # assuming 3rd column is Actual
y_pred = df.iloc[:, 3]   # assuming 4th column is Predicted

mse = mean_squared_error(y_true, y_pred)
rmse = np.sqrt(mse)
mae = mean_absolute_error(y_true, y_pred)
r2 = r2_score(y_true, y_pred)

print("=== A2: Regression Metrics ===")
print(f"MSE: {mse:.4f}")
print(f"RMSE: {rmse:.4f}")
print(f"MAE: {mae:.4f}")
print(f"R² Score: {r2:.4f}")
