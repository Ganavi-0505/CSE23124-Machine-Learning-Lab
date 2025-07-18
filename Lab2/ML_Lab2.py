# A1
print("A1")
#import 
import numpy as np 
import pandas as pd 

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
wed_prices = stock_data[stock_data['Day'] == 'Wed']['Price']
mean_wed = statistics.mean(wed_prices)
print(f"\nMean Price on Wednesdays: {mean_wed:.2f}")
print(f"Difference from Overall Mean: {mean_wed - mprice:.2f}")

# Mean of prices in April
stock_data['Date'] = pd.to_datetime(stock_data['Date'])
aprices = stock_data[stock_data['Date'].dt.month == 4]['Price']
if aprices.empty:
    print("No April data found.")
else:
    mean_april = statistics.mean(aprices)
    print(f"\nMean Price in April: {mean_april:.2f}")
    print(f"Difference from Overall Mean: {mean_april - mprice:.2f}")

# Probability of Loss
chg = stock_data['Chg%'].astype(str).str.replace('%', '', regex=False).astype(float)
prob_loss = (chg < 0).sum() / len(chg)
print(f"\nProbability of Making a Loss: {prob_loss:.2f}")

# Probability of profit on wed
wed_chg = stock_data[stock_data['Day'] == 'Wed']['Chg%'].astype(str).str.replace('%', '', regex=False).astype(float)
if wed_chg.empty:
    print("No Wednesday Chg% \data found.")
else:
    prob_profit_wed = (wed_chg > 0).sum() / len(wed_chg)
    print(f"\nProbability of Profit on Wednesdays: {prob_profit_wed:.2f}")

# Scatter plot
chg_clean = stock_data['Chg%'].astype(str).str.replace('%', '', regex=False).astype(float)
days = stock_data['Day'] 

# Assigning numerical value to days for plotting
unique_days = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri']
day_to_num = {day: i for i, day in enumerate(unique_days)}
day_nums = days.map(day_to_num)

# Plot
plt.figure(figsize=(8, 5))
plt.scatter(day_nums, chg_clean, color='teal', alpha=0.7)
plt.xticks(ticks=range(len(unique_days)), labels=unique_days)
plt.xlabel("Day of the Week")
plt.ylabel("Chg%")
plt.title("IRCTC Stock Chg% vs Day (Raw Data)")
plt.grid(True)
plt.tight_layout()
plt.show()
plt.savefig("chg_percent_vs_day.png")
print("\n✅ Plot saved as 'chg_percent_vs_day.png'")

# A4
print("\n\n")

# Loading data
thyroid_data = pd.read_excel("Lab Session Data.xlsx", sheet_name="thyroid0387_UCI")


# identifying datatypes 
print("\nData Types:")
print(thyroid_data.dtypes)

# Categorical attribute encoding scheme
# Label Encoding if there are 3 or fewer unique values else One-Hot Encoding.
cat_cols = thyroid_data.select_dtypes(include='object').columns.tolist()
for col in cat_cols:
    unique_vals = thyroid_data[col].nunique()

    if unique_vals <= 3:
        encoding = 'Label Encoding'
    else:
        encoding = 'One-Hot Encoding'
    
    print(f"{col}: {encoding}")

# Data range for numeric values
print("\nSummary of Numeric Columns:")
print(thyroid_data.describe())

# Missing values
print("\nMissing Values:")
print(thyroid_data.isnull().sum())

# Outlier detection using IQR [uses median based quartiles which are more stable in skewed data]
print("\nOutliers (IQR method):")
numeric_cols = thyroid_data.select_dtypes(include='number').columns
for col in numeric_cols:
    q1 = thyroid_data[col].quantile(0.25)
    q3 = thyroid_data[col].quantile(0.75)
    iqr = q3 - q1
    outliers = ((thyroid_data[col] < (q1 - 1.5 * iqr)) | (thyroid_data[col] > (q3 + 1.5 * iqr))).sum()
    print(f"{col}: {outliers} outliers")

# Mean, Variance and standard deviation
print("\nVariance and Standard Deviation of Numeric Columns:")
print("\nMean, Variance, and Standard Deviation of Numeric Columns:")
for col in numeric_cols:
    mean = thyroid_data[col].mean()
    var = thyroid_data[col].var()
    std = thyroid_data[col].std()
    print(f"{col}: Mean = {mean:.2f}, Variance = {var:.2f}, Std Dev = {std:.2f}")


