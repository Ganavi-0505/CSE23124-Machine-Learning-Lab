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
s=str(input("Entar a string "))
v,c=count_vc(s)
print(f"Vowels: {v} and Consonants: {c}")

# Question 2: Write a program that accepts 2 matrices A and B as input and returns their product AB. 
# Check if A & B are multipliable if not return error message.
def multiply_matrices(matrix_a: list, matrix_b: list) -> list or str:
    #Returns the product of two matrices if dimensions are compatible.
    if len(matrix_a[0]) != len(matrix_b):
        return "Error: Matrices cannot be multiplied due to dimension mismatch."

    result = []
    for row in matrix_a:
        result_row = []
        for j in range(len(matrix_b[0])):
            element = sum(row[k] * matrix_b[k][j] for k in range(len(matrix_b)))
            result_row.append(element)
        result.append(result_row)
    return result

# Main function
A = [[1, 2], [3, 4]]
B = [[5, 6], [7, 8]]
product = multiply_matrices(A, B)
print("Product of matrices:" if isinstance(product, list) else "Error:", product)


