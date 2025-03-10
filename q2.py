import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
'''
#a
data=input("Enter Number seprated by space:")
list=[int(n) for n in data.split(' ')]
series=pd.Series(list)
print("\nPandas Series:\n",series)
#b
print("Index of series:",series.index)
print("Value of series:",series.values)
#c
data1=input("Enter value of array seprated by space: ")
ls=[int(x) for x in data1.split(' ')]
arr=np.array(ls)
print("Numpy array:",type(arr))
print("Pandas Series:",type(series))
#d
indices=input(f"Enter {len(series)} custom indices seprated by space:").split()
if(len(indices)==len(series)):
    series.index=indices
    print("Updated pandas Series:\n",series)
else:
    print("Custom indexing failed")
#e
sindex=input("\nEnter index to find value: ")
if sindex in series.index:
    print(f"Value at index 'sindex':",series[sindex])
else:
    print("Incorrect Index")
#f
df = pd.read_csv("https://raw.githubusercontent.com/mwaskom/seaborn-data/master/titanic.csv")
print("\nFirst 5 rows of the dataset:\n",df.head())
'''
#g
x=list(map(int,input("Enter x values seprated by space: ").split()))
y=list(map(int,input("Enter y values seprated by space: ").split()))
if len(x)!=len(y):
    print("x and y must have same no. of elements")
else:
    #line plot
    plt.figure(figsize=(6,4))
    plt.plot(x,y,marker='.',linestyle='--',color='r',label="Line Plot")
    plt.xlabel('X-axis')
    plt.ylabel('y-axis')
    plt.title("Line plot")
    plt.legend()
    plt.show()
    #bar chart
    plt.figure(figsize=(6,4))
    plt.bar(x,y,color='g',label="Bar chart")
    plt.xlabel('X-axis')
    plt.ylabel('y-axis')
    plt.title("Bar chart")
    plt.legend()
    plt.grid(axis='y')
    plt.show()
    #Scatter Plot
    plt.figure(figsize=(6,4))
    plt.scatter(x,y,color='b',label="Scatter Plot")
    plt.xlabel('X-axis')
    plt.ylabel('y-axis')
    plt.title("Scatter Plot")
    plt.legend()
    plt.grid(True)
    plt.show()
    #pie Chart
    plt.figure(figsize=(6, 6))
    plt.pie(y, labels=x, autopct='%1.1f%%', colors=['blue', 'green', 'red', 'purple', 'orange'])
    plt.title("Pie Chart")
    plt.show()