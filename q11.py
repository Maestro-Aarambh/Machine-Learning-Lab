import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score

file_path = r"D:\Machine-Learning-Lab\pokemon.csv"

df = pd.read_csv(file_path)

df.columns = df.columns.str.strip()
df.columns = df.columns.str.replace('.', '', regex=False)

print(df.columns)

features = ['hp', 'attack', 'defense', 'sp_attack', 'sp_defense', 'speed']

df_filtered = df[['name', 'type1', 'type2'] + features].dropna()

scaler = StandardScaler()
df_filtered[features] = scaler.fit_transform(df_filtered[features])

k_values = [1, 3, 5, 7, 9]
accuracy_scores = []

def apply_kmeans(k):
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    df_filtered['Cluster'] = kmeans.fit_predict(df_filtered[features])

    if k > 1:
        score = silhouette_score(df_filtered[features], df_filtered['Cluster']) * 100
        accuracy_scores.append(score)
        print(f"Accuracy for k={k}: {score:.2f}%")
    else:
        accuracy_scores.append(None)
        print(f"Accuracy for k={k}: Not applicable")
''''
    plt.figure(figsize=(8, 6))
    plt.scatter(df_filtered['attack'], df_filtered['defense'], c=df_filtered['Cluster'], cmap='viridis', alpha=0.6, edgecolors='k')
    plt.xlabel("Attack")
    plt.ylabel("Defense")
    plt.title(f"K-Means Clustering with k = {k}")
    plt.colorbar(label="Cluster")
    plt.show()
'''
for k in k_values:
    apply_kmeans(k)

plt.figure(figsize=(8, 5))
plt.plot(k_values[1:], accuracy_scores[1:], marker='o', linestyle='-', color='b', label="Accuracy (%)")
plt.xlabel("Number of Clusters (k)")
plt.ylabel("Accuracy (%)")
plt.title("Accuracy vs. Number of Clusters (k)")
plt.xticks(k_values[1:])
plt.grid(True)
plt.legend()
plt.show()
