import numpy as np 
import pandas as pd 

# A1
print("A1")
# Loading Purchase data
f=pd.read_excel("Lab Session Data.xlsx", sheet_name="Purchase data")
# Segregating matrix a and c
A = f[['Candies (#)', 'Mangoes (Kg)', 'Milk Packets (#)']].values
C = f['Payment (Rs)'].values
# dimensionality of vector space
print("Matrix A:\n", A)
print("\n Dimensionality of A:", A.shape[1])
print("\n Number of vectors in A",A.shape[0])

#Finding rank
rankA = np.linalg.matrix_rank(A)
print("\n Rank of Matrix A: ", rankA)

#Finding pseudo inverse
pinvA = np.linalg.pinv(A)
#Finding cost of each item
X=pinvA @ C
print("\n Cost per unit [Candy, Mango, Milk]: ",X)

# A2
#import
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

print("\n \n A2")
# Labelling the customers
labels = ['RICH' if payment > 200 else 'POOR' for payment in C]
# Encoding RICH / POOR as 1/0
le = LabelEncoder()
y = le.fit_transform(labels)
# Split into train/test sets 80:20
xtrain, xtest, ytrain, ytest, ctrain, ctest = train_test_split(
    A, y, f['Customer'], test_size=0.2, random_state=42
)
#Training the logistic regression model
model = LogisticRegression()
model.fit(xtrain,ytrain)
#Testing
ypred = model.predict(xtest)
# Checking accuracy
accu = accuracy_score(ytest, ypred)
print(f"\n Test Accuracy: {accu * 100:.2f}%")
# Display output
for name, pred in zip(ctest, ypred):
    label = le.inverse_transform([pred])[0]
    print(f"\n {name}: Predicted as {label}")

#Alternative
# Label customers based on payment
#labels = ['RICH' if payment > 200 else 'POOR' for payment in C]

# Print classified labels
#for name, label in zip(df['Customer'], labels):
 #   print(f"{name}: {label}")'''


# A3
print("\n\nA3")
#import
import statistics
import matplotlib.pyplot as plt
# Load IRCTC Stock Price sheet
stock_data = pd.read_excel("Lab Session Data.xlsx", sheet_name="IRCTC Stock Price")
# Mean and Variance of Price
prices = stock_data['Price']
mprice = statistics.mean(prices)
vprice = statistics.variance(prices)
print(f"\nMean Price: {mprice:.2f}")
print(f"Variance in Price: {vprice:.2f}")
# Mean of prices on wednesdays
wed_prices = stock_data[stock_data['Day'] == 'Wednesday']['Price']
mean_wed = statistics.mean(wed_prices)
print(f"\nMean Price on Wednesdays: {mean_wed:.2f}")
print(f"Difference from Overall Mean: {mean_wed - mean_price:.2f}")
