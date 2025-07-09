# Question 1: Write a program to count the number of vowels and consonants present in an input string
def count_vc(s:str)->tuple:
    v="aeiouAEIOU"
    cv=0
    cc=0
    for i in s:
        if i in v:
            cv+=1
        else:
            cc+=1
    return cv,cc
# main function
s=str(input("Entar a string "))
v,c=count_vc(s)
print(f"Vowels: {v} and Consonants: {c}")

# Question 2: Write a program that accepts 2 matrices A and B as input and returns their product AB. 
# Check if A & B are multipliable if not return error message.
def matrixmul(a:list, b:list) -> list or str:
    if len(a[0]) != len(b):
        return "Error: Multiplication of these matrices is not possible"
    
    result = []
    for i in a:
        r=[]
        for j in range(len(b[0])):
            for k in range(len(b)):
                p=sum(i[k]*b[k][j])
            r.append(p)
        result.append(r)
    return result
# main function
a=list(input("Enter first matrix "))
b=list(input("Enter second matrix "))
p=matrixmul(a,b)
print("Result",p)

# Question 3: program to find number of common elements between 2 lists of integers
def countc(a:list,b:list)->int:
    s1=set(a)
    s2=set(b)
    r=s1.intersection(s2)
    return len(r)
# main function
a=list(input("Enter first matrix "))
b=list(input("Enter second matrix "))
r=countc(a,b)
print("Result: ",r)

# Question 4: write a program that accepts a matrix as input and returns its transpose
def transpose_matrix(matrix: list) -> list:
    """Returns the transpose of a matrix."""
    rows = len(matrix)
    cols = len(matrix[0])
    transposed = [[matrix[j][i] for j in range(rows)] for i in range(cols)]
    return transposed
# main function
matrix_input = list(input("Enter a matrix"))
transposed = transpose_matrix(matrix_input)
print("Transpose of the matrix:", transposed)

#Question 5: Generate a list of 100 random numbers between 100 and 150. Find the mean median and mode for these numbers 
