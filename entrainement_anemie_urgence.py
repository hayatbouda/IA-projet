import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
import joblib

try:
    df_anemie = pd.read_csv('donnees_anemie.csv')
    print("✅ Données d'anémie chargées.")
except FileNotFoundError:
    print("❌ Erreur: 'donnees_anemie.csv' introuvable. Exécutez ETAPE 1 en premier.")
    exit()

X_anemie = df_anemie[['Hemoglobine', 'MCH', 'MCV']] 

scaler_anemie = StandardScaler()
X_anemie_scaled = scaler_anemie.fit_transform(X_anemie)

y_anemie = df_anemie['Anemie']
y_urgence = df_anemie['Urgence']

X_train_a, X_test_a, y_train_a, y_test_a = train_test_split(X_anemie_scaled, y_anemie, test_size=0.2, random_state=42)
X_train_u, X_test_u, y_train_u, y_test_u = train_test_split(X_anemie_scaled, y_urgence, test_size=0.2, random_state=42)

anemia_model = LogisticRegression(random_state=42)
anemia_model.fit(X_train_a, y_train_a)
print(f"\n✅ Modèle Anémie (LogReg) entraîné. Précision: {accuracy_score(y_test_a, anemia_model.predict(X_test_a)):.4f}")

urgency_model = KNeighborsClassifier(n_neighbors=3)
urgency_model.fit(X_train_u, y_train_u)
print(f"✅ Modèle Urgence (KNN) entraîné. Précision: {accuracy_score(y_test_u, urgency_model.predict(X_test_u)):.4f}")

joblib.dump(scaler_anemie, 'scaler_anemie.pkl')
joblib.dump(anemia_model, 'modele_anemie.pkl')
joblib.dump(urgency_model, 'modele_urgence.pkl')
print("\n✅ Sauvegarde des modèles Anémie et Urgence terminée.")