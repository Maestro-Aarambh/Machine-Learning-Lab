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

features = ['hp', 'attack', 'defense', 'sp_attack', 'sp_defense', 'speed']

df_filtered = df[['name', 'type1', 'type2'] + features].dropna()

scaler = StandardScaler()
df_filtered[features] = scaler.fit_transform(df_filtered[features])

k_values = [1, 3, 5, 7, 9]
accuracy_scores = []

def apply_kmeans(k):
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    df_filtered['Cluster'] = kmeans.fit_predict(df_filtered[features])

    print(f"\nK-Means Clustering with k = {k}")
    for cluster in range(k):
        cluster_pokemon = df_filtered[df_filtered['Cluster'] == cluster]['name'].tolist()
        print(f"Cluster {cluster}: {', '.join(cluster_pokemon[:10])}...")  # Shows first 10 Pokémon per cluster

    if k > 1:
        score = silhouette_score(df_filtered[features], df_filtered['Cluster']) * 100
        accuracy_scores.append(score)
        print(f"\nAccuracy for k={k}: {score:.2f}%")
    else:
        accuracy_scores.append(None)
        print(f"\nAccuracy for k={k}: Not applicable")

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
