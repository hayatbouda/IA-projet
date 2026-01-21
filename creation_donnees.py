import pandas as pd
import numpy as np
from sklearn.datasets import make_classification

print("Création du jeu de données de Transfusion")

N_samples = 748
df_transfusion = pd.DataFrame({
    'Recence (mois)': np.random.randint(0, 75, N_samples),
    'Frequence (nombre)': np.random.randint(1, 60, N_samples),
    'Monetaire (cc)': np.random.randint(250, 15000, N_samples),
    'Temps (mois)': np.random.randint(0, 75, N_samples),
})

df_transfusion['Transfusion'] = np.where(
    (df_transfusion['Frequence (nombre)'] > 10) & (df_transfusion['Recence (mois)'] < 20), 
    1, 
    0
)

df_transfusion.to_csv('donnees_transfusion.csv', index=False)
print("✅ Fichier 'donnees_transfusion.csv' créé avec succès.")

print("\n--- Création du jeu de données Anémie/Urgence ---")

N_samples_a = 1000
df_anemie = pd.DataFrame({
    'Hemoglobine': np.random.uniform(7.0, 17.0, N_samples_a), 
    'MCH': np.random.uniform(20.0, 35.0, N_samples_a), 
    'MCV': np.random.uniform(75.0, 105.0, N_samples_a), 
})

df_anemie['Anemie'] = np.where(df_anemie['Hemoglobine'] < 12.0, 1, 0)

df_anemie['Urgence'] = np.where(df_anemie['Hemoglobine'] < 8.0, 1, 0)

df_anemie.to_csv('donnees_anemie.csv', index=False)
print("✅ Fichier 'donnees_anemie.csv' créé avec succès.")