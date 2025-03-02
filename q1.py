import statistics as stats
import math
import numpy as np
import scipy.linalg as sc
#a
x=float(input("Enter a decimal number "))
y=int(input("Enter an integer number "))
z=int(input("Enter an integer number "))
print("Floor value is ",math.floor(x))
print("Ceil value is ",math.ceil(x))
print("Square root is",math.sqrt(x))
print("Integer Square root is",math.isqrt(y))
print("Greatest Common Divisor is ",math.gcd(z,y))
#b
row=int(input("Enter number of rows "))
col=int(input("Enter number of coloumns "))
arr=[]
for i in range(row):
    while True:
        row=list(map(int,input(f"enter{col} values for row {i+1} ").split()))
        if len(row)==col:
            arr.append(row)
            break
        else:
            print("Invalid input")
arr=np.array(arr)
print("Array:\n",arr)
print("Number of dimensios:",arr.ndim)
print("shape:",arr.shape)
print("size:",arr.size)
print("Row sum:",np.sum(arr,axis=1))
print("coloumn sum:",np.sum(arr,axis=0))
print("mean of elemnts:",np.mean(arr))
print("row sorting:",np.sort(arr,axis=1))
arr1=np.array([0,np.pi/2,np.pi])
print(np.sin(arr1))
#c
det=np.linalg.det(arr)
print("determinant of Matrix arr:",det)
eval,evec=np.linalg.eig(arr)
print("Eigenvalues:",eval)
print("Eigenvectors:",evec)
print("Inverse of arr:\n ",np.linalg.inv(arr))
#d
lst=list(map(int,input("Enter Number seprated by space").split()))
arr2=np.array(lst)
print("2d matrix from list:\n",arr2.reshape(2,3))
print("3d matrix from list:\n",arr2.reshape(2,1,3))
#e
def genrator():
    for i in range(5):
        yield i
gen=genrator()
print(next(gen))
print(next(gen))
print(next(gen))
#f
det_S=sc.det(arr)
print("determinant using scipy:",det_S)
#g
seval,sevec=sc.eig(arr)
print("Eigenvalues in Scipy:",seval)
print("Eigenvectors in scipy:",sevec)