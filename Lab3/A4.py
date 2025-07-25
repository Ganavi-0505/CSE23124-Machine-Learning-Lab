import pandas as pd
from sklearn.model_selection import train_test_split

df = pd.read_excel("lab_3_dataset.xlsx")
classes = df['model'].unique()[:2]
df = df[df['model'].isin(classes)].dropna(axis=1)

y = df['model']
X = df.drop(columns=['model', 'serial_number', 'date', 'datacenter', 'cluster_id'], errors='ignore')

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Save split datasets for reuse
X_train.to_csv("X_train.csv", index=False)
X_test.to_csv("X_test.csv", index=False)
y_train.to_csv("y_train.csv", index=False)
y_test.to_csv("y_test.csv", index=False)
print("Completed")
