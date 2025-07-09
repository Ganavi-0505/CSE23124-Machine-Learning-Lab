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
