import pandas as pd
import numpy as np
from sklearn.datasets import fetch_openml
from scipy import stats
#a
data = {
    'Number': [1, 2, 3, 4, 5],
    'Square': [1, 4, 9, 16, 25]
}
df1 = pd.DataFrame(data)
print("Simple Created DataFrame:\n", df1)

file = r"D:\Machine-Learning-Lab\pokemon.csv"
df = pd.read_csv(file)
print("\nLoaded CSV DataFrame:\n", df.head())

covertype = fetch_openml(name="covertype", version=3, as_frame=True)
print("\nSklearn Dataset (covertype):\n", covertype.frame.head())
#b
mean = df1['Square'].mean()
median = df1['Square'].median()
variance = df1['Square'].var()
std_dev = df1['Square'].std()
mode = stats.mode(df1['Square'], keepdims=True).mode[0]
print("\nStatistics on 'Square' column:")
print(f"Mean: {mean}")
print(f"Median: {median}")
print(f"Mode: {mode}")
print(f"Variance: {variance}")
print(f"Standard Deviation: {std_dev}")
#c
print("\nReshaping Data:")
arr = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
reshaped_arr = arr.reshape(2, 5)
print("Original Array:\n", arr)
print("Reshaped to 2x5:\n", reshaped_arr)

print("\nFiltering Data:")
filtered_df = df1[df1['Square'] > 10]
print("Filtered DataFrame (Square > 10):\n", filtered_df)

print("\nMerging Data:")
data2 = {
    'Number': [1, 2, 3, 4, 5],
    'Cube': [1, 8, 27, 64, 125]
}
df2 = pd.DataFrame(data2)
merged_df = pd.merge(df1, df2, on='Number')
print("Merged DataFrame:\n", merged_df)

data3 = {
    'Number': [1, 2, 3, 4, 5],
    'Square': [1, np.nan, 9, np.nan, 25]
}
df3 = pd.DataFrame(data3)
print("Original DataFrame with NaNs:\n", df3)
df_fill = df3.copy()
df_fill['Square'] = df_fill['Square'].fillna(df_fill['Square'].mean())
print("\nAfter Filling NaNs with Mean:\n", df_fill)

print("\nMin-Max Normalization:")
dfn = df1.copy()
min = dfn['Square'].min()
max = dfn['Square'].max()
dfn['Square_Normalized'] = (dfn['Square'] - min) / (max - min)
print("DataFrame after Min-Max Normalization:\n", dfn)