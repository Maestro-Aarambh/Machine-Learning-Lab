import pandas as pd
import matplotlib as plt
#a
data=input("Enter Number seprated by space:")
list=[int(n) for n in data.split(' ')]
series=pd.Series(list)
print("\nPandas Series:\n",series)