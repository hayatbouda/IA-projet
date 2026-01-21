import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import joblib

try:
    df = pd.read_csv('donnees_transfusion.csv')
    print("✅ Données de transfusion chargées.")
except FileNotFoundError:
    print("❌ Erreur: 'donnees_transfusion.csv' introuvable. Exécutez ETAPE 1 en premier.")
    exit()

X = df.drop('Transfusion', axis=1)
y = df['Transfusion']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler_transfusion = StandardScaler()
X_train_scaled = scaler_transfusion.fit_transform(X_train)
X_test_scaled = scaler_transfusion.transform(X_test)

models = {
    "LogisticRegression": LogisticRegression(random_state=42),
    "KNN": KNeighborsClassifier(n_neighbors=5),
    "DecisionTree": DecisionTreeClassifier(random_state=42)
}

best_accuracy = 0
best_model_name = ""
best_model_instance = None

print("\n--- Entraînement des modèles ML classiques ---")
for name, model in models.items():
    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)
    accuracy = accuracy_score(y_test, y_pred)
    
    if accuracy > best_accuracy:
        best_accuracy = accuracy
        best_model_name = name
        best_model_instance = model
        
    print(f"Modèle {name}: Précision (Accuracy) = {accuracy:.4f}")

print("\n--- Entraînement du Réseau de Neurones ---")
input_dim = X_train_scaled.shape[1]
nn_model = Sequential([
    Dense(16, activation='relu', input_shape=(input_dim,)),
    Dense(8, activation='relu'),
    Dense(1, activation='sigmoid')
])
nn_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
nn_model.fit(X_train_scaled, y_train, epochs=30, batch_size=16, verbose=0)
nn_loss, nn_accuracy = nn_model.evaluate(X_test_scaled, y_test, verbose=0)

if nn_accuracy > best_accuracy:
    best_accuracy = nn_accuracy
    best_model_name = "NeuralNetwork"
    
print(f"Modèle Neural Network: Précision (Accuracy) = {nn_accuracy:.4f}")

joblib.dump(scaler_transfusion, 'scaler_transfusion.pkl')
joblib.dump(best_model_instance, 'modele_transfusion.pkl') 
nn_model.save('modele_transfusion_nn.h5')

print(f"\n✅ Sauvegarde terminée. Meilleur modèle classique: {best_model_name}.")