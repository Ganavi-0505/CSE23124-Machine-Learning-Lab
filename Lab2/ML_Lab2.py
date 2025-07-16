import numpy as np 
import pandas as pd 

# A1
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
