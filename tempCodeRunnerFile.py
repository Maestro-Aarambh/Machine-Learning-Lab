import pandas as pd
import numpy as np
import requests
import matplotlib.pyplot as plt
from PIL import Image
from io import BytesIO
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

file_path = r"D:\Machine-Learning-Lab\pokemon.csv"
df = pd.read_csv(file_path)

df['type1'] = df['type1'].str.lower()
df['type2'] = df['type2'].str.lower()

features = ['hp', 'attack', 'defense', 'sp_attack', 'sp_defense', 'speed']
df_filtered = df[['name', 'type1', 'type2'] + features].dropna().reset_index(drop=True)

scaler = StandardScaler()
df_filtered[features] = scaler.fit_transform(df_filtered[features])

pca = PCA(n_components=2)
pca_features = pca.fit_transform(df_filtered[features])
df_filtered['PCA1'], df_filtered['PCA2'] = pca_features[:, 0], pca_features[:, 1]

kmeans = KMeans(n_clusters=6, random_state=None, n_init=10)
df_filtered['Cluster'] = kmeans.fit_predict(df_filtered[['PCA1', 'PCA2']])

type_chart = {
    'normal': {'weak': ['fighting'], 'resist': [], 'immune': ['ghost']},
    'fire': {'weak': ['water', 'rock', 'ground'], 'resist': ['fire', 'grass', 'ice', 'bug', 'steel', 'fairy'], 'immune': []},
    'water': {'weak': ['electric', 'grass'], 'resist': ['fire', 'water', 'ice', 'steel'], 'immune': []},
    'electric': {'weak': ['ground'], 'resist': ['electric', 'flying', 'steel'], 'immune': []},
    'grass': {'weak': ['fire', 'ice', 'poison', 'flying', 'bug'], 'resist': ['water', 'electric', 'grass', 'ground'], 'immune': []},
    'ice': {'weak': ['fire', 'fighting', 'rock', 'steel'], 'resist': ['ice'], 'immune': []},
    'fighting': {'weak': ['flying', 'psychic', 'fairy'], 'resist': ['bug', 'rock', 'dark'], 'immune': []},
    'poison': {'weak': ['ground', 'psychic'], 'resist': ['grass', 'fighting', 'poison', 'bug', 'fairy'], 'immune': []},
    'ground': {'weak': ['water', 'grass', 'ice'], 'resist': ['poison', 'rock'], 'immune': ['electric']},
    'flying': {'weak': ['electric', 'ice', 'rock'], 'resist': ['grass', 'fighting', 'bug'], 'immune': ['ground']},
    'psychic': {'weak': ['bug', 'ghost', 'dark'], 'resist': ['fighting', 'psychic'], 'immune': []},
    'bug': {'weak': ['fire', 'flying', 'rock'], 'resist': ['grass', 'fighting', 'ground'], 'immune': []},
    'rock': {'weak': ['water', 'grass', 'fighting', 'ground', 'steel'], 'resist': ['normal', 'fire', 'poison', 'flying'], 'immune': []},
    'ghost': {'weak': ['ghost', 'dark'], 'resist': ['poison', 'bug'], 'immune': ['normal', 'fighting']},
    'dragon': {'weak': ['ice', 'dragon', 'fairy'], 'resist': ['fire', 'water', 'electric', 'grass'], 'immune': []},
    'dark': {'weak': ['fighting', 'bug', 'fairy'], 'resist': ['ghost', 'dark'], 'immune': ['psychic']},
    'steel': {'weak': ['fire', 'fighting', 'ground'], 'resist': ['normal', 'grass', 'ice', 'flying', 'psychic', 'bug', 'rock', 'dragon', 'steel', 'fairy'], 'immune': ['poison']},
    'fairy': {'weak': ['poison', 'steel'], 'resist': ['fighting', 'bug', 'dark'], 'immune': ['dragon']}
}

def get_true_weaknesses(type1, type2=None):
    if type1 not in type_chart:
        return "Unknown Type"
    primary_weaknesses = set(type_chart[type1]['weak'])
    primary_resistances = set(type_chart[type1]['resist'])
    primary_immunities = set(type_chart[type1]['immune'])
    if type2 and type2 in type_chart:
        secondary_weaknesses = set(type_chart[type2]['weak'])
        secondary_resistances = set(type_chart[type2]['resist'])
        secondary_immunities = set(type_chart[type2]['immune'])
        true_weaknesses = (primary_weaknesses | secondary_weaknesses)  
        true_weaknesses -= (primary_resistances | secondary_resistances)  
        true_weaknesses -= (primary_immunities | secondary_immunities)  
    else:
        true_weaknesses = primary_weaknesses - primary_resistances - primary_immunities
    return ', '.join(true_weaknesses) if true_weaknesses else "None"

def get_pokemon_image(pokemon_name):
    try:
        url = f"https://pokeapi.co/api/v2/pokemon/{pokemon_name.lower()}"
        response = requests.get(url)
        if response.status_code == 200:
            data = response.json()
            return data['sprites']['front_default']
        else:
            return None
    except:
        return None

def build_pokemon_team(pokemon_type):
    pokemon_type = pokemon_type.lower()
    type_pokemon = df_filtered[(df_filtered['type1'] == pokemon_type) | (df_filtered['type2'] == pokemon_type)]
    if type_pokemon.empty:
        print(f"No Pokémon found with type: {pokemon_type.capitalize()}.")
        return None
    team = []
    for cluster in range(6):
        poke_cluster = type_pokemon[type_pokemon['Cluster'] == cluster]
        if not poke_cluster.empty:
            team.append(poke_cluster.sample(1))  
    if not team:
        print("No suitable Pokémon team found.")
        return None
    team_df = pd.concat(team).reset_index(drop=True)
    print("\nYour Pokémon Team:")
    team_df['Weaknesses'] = team_df.apply(lambda row: get_true_weaknesses(row['type1'], row['type2']), axis=1)
    print(team_df[['name', 'type1', 'type2', 'Weaknesses']])
    fig, axes = plt.subplots(1, len(team_df), figsize=(15, 5))
    if len(team_df) == 1:
        axes = [axes]  
    for i, pokemon in enumerate(team_df['name']):  
        img_url = get_pokemon_image(pokemon)
        if img_url:
            response = requests.get(img_url)
            img = Image.open(BytesIO(response.content))
            axes[i].imshow(img)
        else:
            axes[i].imshow(np.zeros((64, 64, 3)))  
        axes[i].axis('off')
        axes[i].set_title(f"{pokemon}\nWeak: {team_df.loc[i, 'Weaknesses']}", fontsize=9, wrap=True)
    plt.tight_layout()
    plt.show()
    return team_df[['name', 'type1', 'type2', 'Weaknesses']]

pokemon_type = input("Enter a Pokémon type (e.g., Fire, Water, Electric): ").strip()
team = build_pokemon_team(pokemon_type)